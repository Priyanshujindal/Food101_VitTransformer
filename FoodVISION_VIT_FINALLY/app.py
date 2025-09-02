import gradio as gr
from implementation import predict_image

def create_interface():
    with gr.Blocks(title="Food Vision AI") as interface:
        gr.Markdown("# 🍕 Food Vision Transformer 🍕")
        gr.Markdown("## Upload a food image and get AI-powered predictions!")
        gr.Markdown(
            """
            <div style="background:#442a00; color:#ffd79a; padding:12px; border-radius:8px; border:1px solid #996300;">
            ⚠️ <b>Experimental model</b>: Predictions may be inaccurate and can classify <b>non‑food images</b>. Do not use for critical decisions.
            </div>
            """
        )
        with gr.Row():
            with gr.Column():
                input_image=gr.Image(
                    type="pil",
                    label="Uploaded Food Vision Image",
                    height=300
                )
                predict_btn=gr.Button("🔍 Predict", variant="primary")
            with gr.Column():
                output_result=gr.Textbox(label="Prediction Results")
                confidence_slider = gr.Slider(
                    minimum=0, 
                    maximum=1, 
                    value=0, 
                    label="Confidence Score",
                    interactive=False
                )
        predict_btn.click(
            fn=predict_image,
            inputs=input_image,
            outputs=[output_result,confidence_slider]
        )
      
        return interface


if __name__=="__main__":
    app=create_interface()
    app.launch(share=True)
