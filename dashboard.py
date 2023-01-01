from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import model_helper

def run_dashboard(model):
    testing = model_helper.load_testing("testing.csv")
    testing_X = model_helper.get_testing_X_df(testing)
    testing_Y = model_helper.get_testing_Y_df(testing)
    explainer = ClassifierExplainer(model, testing_X, testing_Y)
    ExplainerDashboard(explainer).run()

def main():
    gradientBoostingMachine = model_helper.load_model("gradientBoostingMachine.sav")
    run_dashboard(gradientBoostingMachine)


if __name__ == "__main__":
    main()