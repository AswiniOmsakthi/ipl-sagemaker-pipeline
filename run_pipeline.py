# CI/CD Test - GitHub Actions Trigger - 29 June 2026

import boto3
import sagemaker
import os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.sklearn import SKLearn, SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile


def run_pipeline():

    # ─── Setup ──────────────────────────────────────────────
    region  = os.environ.get("AWS_REGION", "us-east-1")
    role    = os.environ.get("AWS_ROLE_ARN")
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    bucket  = f"sagemaker-ipl-pipeline-{account}"
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    print(f"✅ Region  : {region}")
    print(f"✅ Role    : {role}")
    print(f"✅ Bucket  : {bucket}")

    # ─── Step 1: Data Preparation ───────────────────────────
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        sagemaker_session=session
    )

    data_prep_step = ProcessingStep(
        name="DataPreparation",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/data/ipl_final.csv",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/processed/train"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/processed/test"
            )
        ],
        code="pipeline/preprocessing.py"
    )

    # ─── Step 2: Model Training ─────────────────────────────
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="pipeline",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        use_spot_instances=True,        # ✅ Spot = 90% savings
        max_run=3600,
        max_wait=7200,
        sagemaker_session=session
    )

    train_step = TrainingStep(
        name="ModelTraining",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=data_prep_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    # ─── Step 3: Model Evaluation ───────────────────────────
    evaluator = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        sagemaker_session=session
    )

    # PropertyFile to read metrics from evaluation.json
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    eval_step = ProcessingStep(
        name="ModelEvaluation",
        processor=evaluator,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=data_prep_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/evaluation"
            )
        ],
        code="pipeline/evaluate.py",
        property_files=[evaluation_report]  # ✅ Register property file
    )

    # ─── Step 4: Model Registration ─────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"s3://{bucket}/evaluation/evaluation.json",
            content_type="application/json"
        )
    )

    register_step = RegisterModel(
        name="ModelRegistration",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="IPLMatchPredictionGroup",
        approval_status="PendingManualApproval",
        model_metrics=model_metrics
    )

    # ─── Step 5: Quality Gate ───────────────────────────────
    # Read accuracy from evaluation.json
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value"  # ✅ Read from report
        ),
        right=0.75   # ✅ Threshold: 75% accuracy required
    )

    condition_step = ConditionStep(
        name="QualityGate",
        conditions=[accuracy_condition],
        if_steps=[register_step],   # ✅ accuracy >= 0.75 → Register
        else_steps=[]               # ❌ accuracy < 0.75  → Stop
    )

    # ─── Build & Run Pipeline ───────────────────────────────
    pipeline = Pipeline(
        name="IPLMatchPredictionPipeline",
        steps=[
            data_prep_step,
            train_step,
            eval_step,
            condition_step          # ✅ QualityGate + Registration inside
        ],
        sagemaker_session=session
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()

    print(f"✅ Pipeline Started!")
    print(f"✅ Execution ARN : {execution.arn}")

    # Wait and print results
    execution.wait()
    print("\n✅ Pipeline Completed!")
    print("\n📊 Step Results:")

    for step in execution.list_steps():
        print(f"   {step['StepName']:30} : {step['StepStatus']}")


if __name__ == "__main__":
    run_pipeline()