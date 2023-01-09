from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import model_helper
from os.path import exists

def save_dashboard(model):
    testing = model_helper.load_testing("testing.csv")
    testing_X = model_helper.get_testing_X_df(testing)
    testing_Y = model_helper.get_testing_Y_df(testing)

    explainer = ClassifierExplainer(model, testing_X, testing_Y)

    db = ExplainerDashboard(explainer, title="Breast Cancer Classification - Gradient Boosting Dashboard", shap_interaction=False)
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)

def create_app():
    gradientBoostingMachine = model_helper.load_model("gradientBoostingMachine.sav")
    save_dashboard(gradientBoostingMachine)
    db = ExplainerDashboard.from_config("dashboard.yaml")
    app = db.flask_server()
    return app