import asyncio
from flotta.core.distributions import Collect
from flotta.core.model_operations import Aggregation, Train, TrainTest
from flotta.core.models import FederatedRandomForestClassifier, StrategyRandomForestClassifier
from flotta.core.steps import Finalize, Parallel
from flotta.core.transformers import FederatedSplitter
from flotta.workbench import Context, Artifact
import matplotlib.pyplot as plt

# Define the asynchronous function
async def submit_training():
    server_url = "http://localhost:1456"
    project_token = "58981bcbab77ef4b8e01207134c38873e0936a9ab88cd76b243a2e2c85390b94"

    # Create the context
    ctx = Context(server_url)

    # Load a project
    project = ctx.project(project_token)

    # Create a Federated model
    model = FederatedRandomForestClassifier(
        n_estimators=10,
        strategy=StrategyRandomForestClassifier.MERGE,
    )

    label = "label"

    # Describe how to distribute the work and how to train the model
    steps = [
        Parallel(
            TrainTest(
                query=project.extract().add(
                    FederatedSplitter(
                        random_state=42,
                        test_percentage=0.2,
                        label=label,
                    )
                ),
                trainer=Train(model=model),
                model=model,
            ),
            Collect(),
        ),
        Finalize(
            Aggregation(model=model),
        ),
    ]

    # Submit the artifact asynchronously
    artifact: Artifact = await ctx.submit(project, steps)

    return artifact

# Function to monitor progress asynchronously
async def monitor_progress(artifact):
    while not artifact.is_complete():
        # Fetch and print progress while the task is still running
        progress = await artifact.get_training_progress()
        print("Training Progress:", progress)
        await asyncio.sleep(5)  # Wait for 5 seconds before checking progress again
    print("Training is complete!")

# Run the asynchronous function
async def main():
    server_url = "http://localhost:1456"
    project_token = "58981bcbab77ef4b8e01207134c38873e0936a9ab88cd76b243a2e2c85390b94"
    
    # Create the context
    ctx = Context(server_url)
    # Load a project
    project = ctx.project(project_token)

    artifact = await submit_training()  # Submit training job
    await monitor_progress(artifact)  # Monitor progress while the training job is running
    print("Final artifact:", artifact)  # After completion, print the artifact

    # If the accuracy is available in the artifact, plot it
    try:
        accuracy = artifact.get_accuracy()  # This depends on Flotta's actual API
        plt.plot(accuracy)
        plt.title('Model Accuracy Over Training Rounds')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.show()
    except AttributeError:
        print("Accuracy not available in the artifact")

    # Perform inference directly using the artifact
    inference_data = project.extract().add(FederatedSplitter(test_percentage=0.2, label='label'))
    predictions = ctx.predict(artifact, inference_data)
    print("Predictions:", predictions)

# Start the asyncio event loop
asyncio.run(main())
