import boto3
import sagemaker
import os
import json
import time


def deploy_model():

    # ─── Setup ──────────────────────────────────────────────
    region  = os.environ.get("AWS_REGION", "us-east-1")
    account = boto3.client("sts", region_name=region)\
                   .get_caller_identity()["Account"]
    bucket  = f"sagemaker-ipl-pipeline-{account}"

    sm_client = boto3.client("sagemaker",         region_name=region)
    s3_client = boto3.client("s3",                region_name=region)
    runtime   = boto3.client("sagemaker-runtime", region_name=region)

    print(f"✅ Region  : {region}")
    print(f"✅ Account : {account}")

    # ─── Read Infra Outputs from S3 ─────────────────────────
    try:
        response  = s3_client.get_object(
            Bucket=bucket,
            Key="infra/infra_outputs.json"
        )
        infra     = json.loads(response["Body"].read())
        role      = infra["role_arn"]
        domain_id = infra["domain_id"]
        bucket    = infra["bucket_name"]

        print(f"✅ Infra outputs loaded")
        print(f"✅ Domain ID : {domain_id}")
        print(f"✅ Role ARN  : {role}")
        print(f"✅ Bucket    : {bucket}")

    except Exception as e:
        print(f"⚠️ S3 read failed : {str(e)}")
        role = os.environ.get("AWS_ROLE_ARN")
        print(f"✅ Fallback Role  : {role}")

    # ─── Get Latest Approved Model ──────────────────────────
    try:
        model_packages = sm_client.list_model_packages(
            ModelPackageGroupName="IPLMatchPredictionGroup-New",  # ✅ New group
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )

        if not model_packages["ModelPackageSummaryList"]:
            raise Exception(
                "No approved model found in IPLMatchPredictionGroup-New! "
                "Please approve model first."
            )

        model_package_arn = model_packages[
            "ModelPackageSummaryList"
        ][0]["ModelPackageArn"]

        model_version = model_packages[
            "ModelPackageSummaryList"
        ][0]["ModelPackageVersion"]

        print(f"✅ Model Package ARN     : {model_package_arn}")
        print(f"✅ Model Package Version : {model_version}")

    except Exception as e:
        print(f"❌ Model fetch error: {str(e)}")
        raise

    # ─── Create SageMaker Session ───────────────────────────
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=region),
        sagemaker_client=sm_client,
        default_bucket=bucket
    )

    # ─── Create Model ────────────────────────────────────────
    model_name    = f"ipl-match-predictor-{int(time.time())}"
    endpoint_name = "ipl-match-predictor-endpoint"

    try:
        model = sagemaker.ModelPackage(
            role=role,
            model_package_arn=model_package_arn,
            sagemaker_session=session,
            name=model_name
        )
        print(f"✅ Model object created : {model_name}")

    except Exception as e:
        print(f"❌ Model creation error: {str(e)}")
        raise

    # ─── Check if Endpoint Exists ───────────────────────────
    try:
        existing        = sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        endpoint_status = existing["EndpointStatus"]
        print(f"✅ Endpoint exists : {endpoint_name}")
        print(f"✅ Status          : {endpoint_status}")

        # Create new endpoint config
        config_name = f"ipl-endpoint-config-{int(time.time())}"

        sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName"         : "AllTraffic",
                    "ModelName"           : model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType"        : "ml.m5.large",
                    "InitialVariantWeight": 1
                }
            ]
        )

        # Update existing endpoint
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"✅ Endpoint update started!")

    except sm_client.exceptions.ClientError:
        # Endpoint does not exist — create new
        print(f"⏳ Creating new endpoint: {endpoint_name}")

        model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.large",
            endpoint_name=endpoint_name,
            wait=False
        )
        print(f"✅ Endpoint creation started!")

    # ─── Wait for Endpoint InService ────────────────────────
    print(f"⏳ Waiting for endpoint to be InService...")

    while True:
        status = sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )["EndpointStatus"]

        print(f"   Status: {status}")

        if status == "InService":
            break
        elif status in ["Failed", "OutOfService"]:
            raise Exception(f"Endpoint failed: {status}")

        time.sleep(30)

    print(f"✅ Endpoint is InService!")

    # ─── Test Endpoint ───────────────────────────────────────
    print(f"\n⏳ Testing endpoint...")

    # team1_won_toss, toss_bat, win_by_runs, win_by_wickets
    test_cases = [
        ("1,1,0,6",   "Team won toss + bat + won by 6 wickets"),
        ("0,0,0,0",   "Team lost toss + field + no result"),
        ("1,0,25,0",  "Team won toss + field + won by 25 runs"),
    ]

    for test_input, description in test_cases:
        response   = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=test_input
        )
        prediction = response["Body"].read().decode("utf-8").strip()
        result     = "WIN ✅" if prediction == "1" else "LOSS ❌"

        print(f"   Input      : {test_input}")
        print(f"   Scenario   : {description}")
        print(f"   Prediction : {result}")
        print()

    # ─── Save Deploy Outputs to S3 ──────────────────────────
    deploy_outputs = {
        "endpoint_name"    : endpoint_name,
        "model_name"       : model_name,
        "model_package_arn": model_package_arn,
        "model_version"    : model_version,
        "group_name"       : "IPLMatchPredictionGroup-New",
        "domain_id"        : domain_id,
        "status"           : "InService",
        "region"           : region
    }

    s3_client.put_object(
        Bucket=bucket,
        Key="deploy/deploy_outputs.json",
        Body=json.dumps(deploy_outputs, indent=2)
    )

    print("\n" + "="*50)
    print("✅ DEPLOYMENT COMPLETE!")
    print(f"   Endpoint    : {endpoint_name}")
    print(f"   Model       : {model_name}")
    print(f"   Version     : {model_version}")
    print(f"   Group       : IPLMatchPredictionGroup-New")
    print(f"   Status      : InService ✅")
    print(f"   Domain      : {domain_id}")
    print("="*50)

    return deploy_outputs


if __name__ == "__main__":
    deploy_model()