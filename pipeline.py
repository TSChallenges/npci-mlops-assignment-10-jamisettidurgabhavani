from kfp.dsl import component, pipeline, Input, Output, Dataset, Model
import kfp

# Pipeline Component-1: Data Loading & Processing
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def load_churn_data(drop_missing_vals: bool, churn_dataset: Output[Dataset]):
    import pandas as pd
    # Load Customer Churn dataset
    df = pd.read_csv("https://raw.githubusercontent.com/MLOPS-test/test-scripts/refs/heads/main/mlops-ast10/Churn_Modeling.csv")

    # Define target variable and features
    X = df[["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"]].copy()
    y = df[["Exited"]]

    # Handling category labels present in `Geography` and `Gender` columns
    # Create dictionaries to map categorical values to numberic labels. OR Use LabelEncoder
    geography_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
    gender_mapping = {'Female': 0, 'Male': 1}

    # Map categorical values to numbers using respective dictionaries
    X['Geography'] = X['Geography'].map(geography_mapping)
    X['Gender'] = X['Gender'].map(gender_mapping)

    transformed_df = X.copy()
    transformed_df['Exited'] = y

    if drop_missing_vals:
        transformed_df = transformed_df.dropna()

    with open(churn_dataset.path, 'w') as file:
        transformed_df.to_csv(file, index=False)


# Pipeline Component-2: Train-Test Split
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def train_test_split_churn(
    input_churn_dataset: Input[Dataset],
    X_train: Output[Dataset],
    X_test: Output[Dataset],
    y_train: Output[Dataset],
    y_test: Output[Dataset],
    test_size: float,
    random_state: int,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # YOUR CODE HERE to split the dataset into training & testing set and save them as CSV files
    df = pd.read_csv(input_churn_dataset.path)
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_tr.to_csv(X_train.path, index=False)
    X_te.to_csv(X_test.path, index=False)
    y_tr.to_csv(y_train.path, index=False)
    y_te.to_csv(y_test.path, index=False)




# Pipeline Component-3: Model Training
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def train_churn_model(
    X_train: Input[Dataset],
    y_train: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int,
    random_state: int,
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    # YOUR CODE HERE to load train the model on training set, and save the model
    X = pd.read_csv(X_train.path)
    y = pd.read_csv(y_train.path).values.ravel()

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)

    with open(model_output.path, 'wb') as f:
        pickle.dump(model, f)




# Pipeline Component-4: Model Evaluation
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def evaluate_churn_model(
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    model_path: Input[Model],
    metrics_output: Output[Dataset]
):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pickle

    # YOUR CODE HERE to check the model performance using different metrics and save them
    X = pd.read_csv(X_test.path)
    y_true = pd.read_csv(y_test.path).values.ravel()

    with open(model_path.path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_output.path, index=False)



# Pipeline Definition
@pipeline(name="customer-churn-pipeline", description="Pipeline for Customer Churn Prediction")
def customer_churn_pipeline(
    drop_missing_vals: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
):
    # YOUR CODE HERE to connect the pipeline components and direct their inputs and outputs
    load_op = load_churn_data(drop_missing_vals=drop_missing_vals)

    split_op = train_test_split_churn(
        input_churn_dataset=load_op.outputs["churn_dataset"],
        test_size=test_size,
        random_state=random_state
    )

    train_op = train_churn_model(
        X_train=split_op.outputs["X_train"],
        y_train=split_op.outputs["y_train"],
        n_estimators=n_estimators,
        random_state=random_state
    )

    eval_op = evaluate_churn_model(
        X_test=split_op.outputs["X_test"],
        y_test=split_op.outputs["y_test"],
        model_path=train_op.outputs["model_output"]
    )




# Compile Pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(customer_churn_pipeline, 'customer_churn_pipeline_v1.yaml')


