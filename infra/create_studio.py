import boto3
import json
import os
import time


def create_sagemaker_studio():

    # ─── Setup ──────────────────────────────────────────────
    region  = os.environ.get("AWS_REGION", "us-east-1")
    account = boto3.client("sts", region_name=region)\
                   .get_caller_identity()["Account"]

    sm_client  = boto3.client("sagemaker", region_name=region)
    iam_client = boto3.client("iam",       region_name=region)
    s3_client  = boto3.client("s3",        region_name=region)
    ec2_client = boto3.client("ec2",       region_name=region)

    domain_name  = "ipl-sagemaker-studio"
    bucket_name  = f"sagemaker-ipl-pipeline-{account}"
    role_name    = "SageMakerStudioExecutionRole"
    profile_name = "ipl-user"

    print(f"✅ Region  : {region}")
    print(f"✅ Account : {account}")
    print(f"✅ Bucket  : {bucket_name}")

    # ─── Step 1: Create S3 Bucket ───────────────────────────
    try:
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    "LocationConstraint": region
                }
            )
        print(f"✅ S3 Bucket created : {bucket_name}")
    except Exception as e:
        print(f"✅ S3 Bucket note    : {str(e)}")

    # ─── Step 2: Create SageMaker Execution Role ────────────
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker Studio Execution Role"
        )
        role_arn = role["Role"]["Arn"]

        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        )
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
        )

        print(f"✅ IAM Role created  : {role_arn}")
        time.sleep(10)

    except iam_client.exceptions.EntityAlreadyExistsException:
        role_arn = f"arn:aws:iam::{account}:role/{role_name}"
        print(f"✅ IAM Role exists   : {role_arn}")

    # ─── Step 3: Get Default VPC ────────────────────────────
    vpcs = ec2_client.describe_vpcs(
        Filters=[{"Name": "isDefault", "Values": ["true"]}]
    )
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    subnets = ec2_client.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
    )
    subnet_ids = [s["SubnetId"] for s in subnets["Subnets"]][:2]

    print(f"✅ VPC      : {vpc_id}")
    print(f"✅ Subnets  : {subnet_ids}")

    # ─── Step 4: Create SageMaker Studio Domain ─────────────
    try:
        existing      = sm_client.list_domains()
        domain_exists = any(
            d["DomainName"] == domain_name
            for d in existing["Domains"]
        )

        if domain_exists:
            domain_id = next(
                d["DomainId"]
                for d in existing["Domains"]
                if d["DomainName"] == domain_name
            )
            print(f"✅ Studio exists : {domain_id}")

        else:
            response = sm_client.create_domain(
                DomainName=domain_name,
                AuthMode="IAM",
                DefaultUserSettings={
                    "ExecutionRole": role_arn,
                    "JupyterServerAppSettings": {
                        "DefaultResourceSpec": {
                            "InstanceType": "system"
                        }
                    },
                    "KernelGatewayAppSettings": {
                        "DefaultResourceSpec": {
                            "InstanceType": "ml.t3.medium"
                        }
                    }
                },
                SubnetIds=subnet_ids,
                VpcId=vpc_id,
                Tags=[
                    {"Key": "Project",     "Value": "IPL-Pipeline"},
                    {"Key": "Environment", "Value": "dev"}
                ]
            )

            domain_id = response["DomainArn"].split("/")[-1]
            print(f"✅ Studio creating : {domain_id}")

            # Wait for domain to be InService
            print("⏳ Waiting for Studio to be InService...")
            while True:
                status = sm_client.describe_domain(
                    DomainId=domain_id
                )["Status"]
                print(f"   Status: {status}")
                if status == "InService":
                    break
                elif status == "Failed":
                    raise Exception("Domain creation failed!")
                time.sleep(30)

            print(f"✅ Studio active : {domain_id}")

    except Exception as e:
        print(f"❌ Domain error: {str(e)}")
        raise

    # ─── Step 5: Create User Profile ────────────────────────
    try:
        sm_client.create_user_profile(
            DomainId=domain_id,
            UserProfileName=profile_name,
            UserSettings={"ExecutionRole": role_arn}
        )
        print(f"✅ User Profile created : {profile_name}")
    except sm_client.exceptions.ResourceInUse:
        print(f"✅ User Profile exists  : {profile_name}")

    # ─── Step 6: Save Outputs to S3 ─────────────────────────
    outputs = {
        "domain_id"   : domain_id,
        "domain_name" : domain_name,
        "role_arn"    : role_arn,
        "bucket_name" : bucket_name,
        "profile_name": profile_name,
        "region"      : region,
        "account"     : account
    }

    # Save locally
    with open("infra_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)

    # Save to S3 so Pipeline 2 and 3 can read it
    s3_client.put_object(
        Bucket=bucket_name,
        Key="infra/infra_outputs.json",
        Body=json.dumps(outputs, indent=2)
    )

    print("\n" + "="*50)
    print("✅ INFRA SETUP COMPLETE!")
    print(f"   Domain ID    : {domain_id}")
    print(f"   Domain Name  : {domain_name}")
    print(f"   Role ARN     : {role_arn}")
    print(f"   S3 Bucket    : {bucket_name}")
    print(f"   User Profile : {profile_name}")
    print(f"   Outputs S3   : s3://{bucket_name}/infra/infra_outputs.json")
    print("="*50)

    return outputs


if __name__ == "__main__":
    create_sagemaker_studio()