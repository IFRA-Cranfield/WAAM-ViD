import gradio as gr
import videoloader
import dimensional_analysis


def gradio_wrapper(video_input_path, csv_input_obj):
    """
    Wrapper between Gradio and the video processing logic.

    Args:
        video_input_path (str): Path to the uploaded video file.
        csv_input_obj (gradio.File): Optional CSV file (currently ignored).

    Returns:
        str or None: Path to the processed video file, or None if failed.
    """
    if csv_input_obj:
        da = dimensional_analysis.Dimensional_Analysis()
        predict_width_list = da.evaluation(video_input_path, csv_input_obj)

    vl = videoloader.Videoloader()

    output_video_path = vl.video_processing(video_input_path, predict_width_list)

    return output_video_path

demo = gr.Interface(
    fn=gradio_wrapper,
    inputs=[
        gr.Video(label="Upload Input Video"),
        gr.File(label="Upload CSV File (Optional)", file_types=['.csv'])
    ],
    outputs=gr.Video(label="Processed Video Output with Segmentation Overlay"),
    title="Video Segmentation Processor",
    description="Upload a video to apply DeepLabV3 segmentation and overlay the mask."
)


if __name__ == "__main__":
    demo.launch(debug=True)