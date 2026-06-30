import boto3
import json
import os
import time
import zipfile
import tempfile


def setup_auto_deploy_trigger():

    region  = os.environ.get("AWS_REGION", "us-east-1")
    account = boto3.client("sts", region_name=region)\
                   .get_caller_identity()["Account"]

    lambda_client = boto3.client("lambda", region_name=region)
    events_client = boto3.client("events", region_name=region)
    iam_client    = boto3.client("iam",    region_name=region)

    function_name = "trigger-deploy-pipeline"
    rule_name     = "model-approved-rule"

    print(f"✅ Setting up auto deploy trigger...")
    print(f"✅ Account : {account}")
    print(f"✅ Region  : {region}")

    # ─── Step 1: Create Lambda Role ─────────────────────────
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        role = iam_client.create_role(
            RoleName="LambdaDeployTriggerRole",
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Lambda role to trigger GitHub Actions"
        )
        role_arn = role["Role"]["Arn"]

        iam_client.attach_role_policy(
            RoleName="LambdaDeployTriggerRole",
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )
        iam_client.attach_role_policy(
            RoleName="LambdaDeployTriggerRole",
            PolicyArn="arn:aws:iam::aws:policy/SecretsManagerReadWrite"
        )

        print(f"✅ Lambda Role created : {role_arn}")
        time.sleep(10)

    except iam_client.exceptions.EntityAlreadyExistsException:
        role_arn = f"arn:aws:iam::{account}:role/LambdaDeployTriggerRole"
        print(f"✅ Lambda Role exists  : {role_arn}")

    # ─── Step 2: Create Lambda Function ─────────────────────
    lambda_code = '''
import json
import urllib.request
import boto3

def lambda_handler(event, context):
    print(f"Event received: {json.dumps(event)}")

    detail = event.get("detail", {})
    status = detail.get("ModelApprovalStatus", "")

    print(f"Model Approval Status: {status}")

    if status != "Approved":
        print("Model not approved - skipping deploy")
        return {"statusCode": 200, "body": "Not approved"}

    sm_client = boto3.client("secretsmanager")
    secret    = sm_client.get_secret_value(SecretId="github-token")

    # ✅ Token is stored as PLAINTEXT, not JSON
    token = secret["SecretString"]

    repo = "AswiniOmsakthi/ipl-sagemaker-pipeline"
    url  = f"https://api.github.com/repos/{repo}/actions/workflows/deploy-pipeline.yml/dispatches"

    payload = json.dumps({"ref": "master"}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            print(f"Pipeline 3 triggered! Status: {response.status}")
            return {"statusCode": 200, "body": "Pipeline 3 triggered successfully!"}
    except Exception as e:
        print(f"Failed to trigger: {str(e)}")
        raise
'''

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("lambda_function.py", lambda_code)

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    try:
        lambda_client.create_function(
            FunctionName=function_name,
            Runtime="python3.10",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Description="Trigger GitHub Actions Pipeline 3 on model approval",
            Timeout=30
        )
        print(f"✅ Lambda created : {function_name}")

    except lambda_client.exceptions.ResourceConflictException:
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_bytes
        )
        print(f"✅ Lambda updated : {function_name}")

    time.sleep(5)

    lambda_arn = lambda_client.get_function(
        FunctionName=function_name
    )["Configuration"]["FunctionArn"]

    # ─── Step 3: Create EventBridge Rule ────────────────────
    rule = events_client.put_rule(
        Name=rule_name,
        EventPattern=json.dumps({
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Model Package State Change"],
            "detail": {
                "ModelApprovalStatus": ["Approved"],
                "ModelPackageGroupName": ["IPLMatchPredictionGroup-New"]
            }
        }),
        State="ENABLED",
        Description="Trigger deploy pipeline when model approved"
    )
    rule_arn = rule["RuleArn"]
    print(f"✅ EventBridge Rule created : {rule_name}")

    # ─── Step 4: Add Lambda Permission ──────────────────────
    try:
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId="EventBridgeInvoke",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn
        )
        print(f"✅ Lambda permission added")
    except lambda_client.exceptions.ResourceConflictException:
        print(f"✅ Lambda permission already exists")

    # ─── Step 5: Add Lambda as Target ───────────────────────
    events_client.put_targets(
        Rule=rule_name,
        Targets=[{"Id": "DeployPipelineTrigger", "Arn": lambda_arn}]
    )

    print("\n" + "="*50)
    print("✅ AUTO DEPLOY TRIGGER SETUP COMPLETE!")
    print(f"   Lambda     : {function_name}")
    print(f"   Rule       : {rule_name}")
    print(f"   Watches    : IPLMatchPredictionGroup-New")
    print(f"   On Approve : Triggers Pipeline 3 ✅")
    print("="*50)


if __name__ == "__main__":
    setup_auto_deploy_trigger()