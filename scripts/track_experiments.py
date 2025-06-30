import mlflow

def list_experiments():
    for exp in mlflow.search_experiments():
        print(f"Name: {exp.name}, ID: {exp.experiment_id}")

if __name__ == "__main__":
    list_experiments()