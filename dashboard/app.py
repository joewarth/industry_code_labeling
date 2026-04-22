from shiny import App, reactive, render, ui
from api_client import predict_naics


app_ui = ui.page_fluid(
    ui.h2("NAICS Industry Code Predictor"),
    ui.p("Enter a company description to predict the 2022 NAICS hierarchy."),
    ui.input_text_area(
        "company_description",
        "Company description",
        rows=8,
        placeholder="Example: Commercial roofing contractor specializing in industrial and warehouse roofing systems.",
    ),
    ui.input_action_button("go", "Predict"),
    ui.hr(),
    ui.output_ui("prediction_panel"),
)


def server(input, output, session):
    @reactive.calc
    @reactive.event(input.go)
    def prediction():
        desc = input.company_description().strip()
        if not desc:
            return {"error": "Please enter a company description."}
        try:
            return predict_naics(desc)
        except Exception as e:
            return {"error": str(e)}

    @output
    @render.ui
    def prediction_panel():
        result = prediction()

        if "error" in result:
            return ui.div(
                ui.h4("Error"),
                ui.p(result["error"]),
            )

        return ui.div(
            ui.h4("Predicted hierarchy"),
            ui.tags.ul(
                ui.tags.li(f"2-digit: {result['pred_y2']}"),
                ui.tags.li(f"3-digit: {result['pred_y3']}"),
                ui.tags.li(f"4-digit: {result['pred_y4']}"),
                ui.tags.li(f"5-digit: {result['pred_y5']}"),
                ui.tags.li(f"6-digit: {result['pred_y6']}"),
            ),
            ui.h4("Confidence"),
            ui.p(f"6-digit confidence: {result['pred_prob_y6']:.4f}"),
            ui.h4("Top 5 6-digit predictions"),
            ui.tags.ol(
                *[
                    ui.tags.li(f"{item['code']} ({item['prob']:.4f})")
                    for item in result["pred_top5_y6"]
                ]
            ),
        )


app = App(app_ui, server)