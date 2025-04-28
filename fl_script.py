from flotta.core.distributions import Collect
from flotta.core.model_operations import Aggregation, Train, TrainTest
from flotta.core.models import FederatedRandomForestClassifier, StrategyRandomForestClassifier
from flotta.core.steps import Finalize, Parallel
from flotta.core.transformers import FederatedSplitter
from flotta.workbench import Context, Artifact

server_url = "http://localhost:1456"
project_token = "58981bcbab77ef4b8e01207134c38873e0936a9ab88cd76b243a2e2c85390b94"

# create the context
ctx = Context(server_url)

# load a project
project = ctx.project(project_token)
# create a Federated model
model = FederatedRandomForestClassifier(
    n_estimators=10,
    strategy=StrategyRandomForestClassifier.MERGE,
)

label = "label"

# describe how to distribute the work and how to train teh model
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

# submit Artifact
artifact: Artifact = ctx.submit(project, steps)