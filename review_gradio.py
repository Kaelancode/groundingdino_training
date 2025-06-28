import os
import csv
import gradio as gr
from PIL import Image, ImageDraw

# Modify these paths
images_dir = "/home/kaelan/Projects/Github/Grounding-Dino-FineTuning/water/ProcessedDataSet/multi_annot_images"
labels_dir = "/home/kaelan/Projects/Github/Grounding-Dino-FineTuning/water/ProcessedDataSet/multi_annot_labels"
output_csv = "/home/kaelan/Projects/Github/Grounding-Dino-FineTuning/water/ProcessedDataSet/accepted_images.csv"

image_files = sorted([
    f for f in os.listdir(images_dir) if f.endswith(".jpg")
])
accepted_data = []
index = 0

def label_id_to_name(label_id):
    return f"class{label_id}"

def draw_annotations(image_path, label_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            label_id, cx_n, cy_n, w_n, h_n = map(float, parts)
            label_name = label_id_to_name(int(label_id))

            bbox_width = w_n * img_width
            bbox_height = h_n * img_height
            bbox_x = (cx_n * img_width) - (bbox_width / 2)
            bbox_y = (cy_n * img_height) - (bbox_height / 2)

            x0, y0 = bbox_x, bbox_y
            x1, y1 = bbox_x + bbox_width, bbox_y + bbox_height

            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            draw.text((x0, y0), label_name, fill="red")
    return img

def get_next(index_state):
    if index_state >= len(image_files):
        return None, gr.update(visible=False), "✅ Done reviewing all images."

    img_file = image_files[index_state]
    label_file = img_file.replace(".jpg", ".txt")

    img_path = os.path.join(images_dir, img_file)
    lbl_path = os.path.join(labels_dir, label_file)

    annotated_img = draw_annotations(img_path, lbl_path)
    return annotated_img, gr.update(visible=True), f"Image {index_state + 1} of {len(image_files)}"

def record_decision(decision, index_state):
    if decision == "yes":
        img_file = image_files[index_state]
        label_file = img_file.replace(".jpg", ".txt")
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, label_file)
        img = Image.open(img_path)
        width, height = img.size

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                label_id, cx_n, cy_n, w_n, h_n = map(float, parts)
                label_name = label_id_to_name(int(label_id))

                bbox_width = w_n * width
                bbox_height = h_n * height
                bbox_x = (cx_n * width) - (bbox_width / 2)
                bbox_y = (cy_n * height) - (bbox_height / 2)

                accepted_data.append([
                    label_name,
                    round(bbox_x, 2),
                    round(bbox_y, 2),
                    round(bbox_width, 2),
                    round(bbox_height, 2),
                    img_file,
                    width,
                    height
                ])
                break
    return index_state + 1

def on_click(decision, index_state):
    next_index = record_decision(decision, index_state)
    return (*get_next(next_index), next_index)

# def export_csv():
#     with open(output_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             'label_name', 'bbox_x', 'bbox_y', 'bbox_width',
#             'bbox_height', 'image_name', 'width', 'height'
#         ])
#         writer.writerows(accepted_data)
#     return f"✅ CSV saved to {output_csv}"

def export_csv():
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'label_name', 'bbox_x', 'bbox_y', 'bbox_width',
                'bbox_height', 'image_name', 'width', 'height'
            ])
        writer.writerows(accepted_data)
    accepted_data.clear()  # Reset after saving
    return f"✅ Appended to {output_csv} and cleared memory"

with gr.Blocks() as demo:
    index_state = gr.State(0)

    with gr.Row():
        image_display = gr.Image(type="pil", label="Annotated Image")
        info_text = gr.Markdown()

    with gr.Row(visible=True) as button_row:
        yes_btn = gr.Button("✅ Yes")
        no_btn = gr.Button("❌ No")

    done_btn = gr.Button("💾 Save Accepted to CSV")
    
    yes_btn.click(
    fn=lambda index: on_click("yes", index),
    inputs=[index_state],
    outputs=[image_display, button_row, info_text, index_state]
    )

    no_btn.click(
        fn=lambda index: on_click("no", index),
        inputs=[index_state],
        outputs=[image_display, button_row, info_text, index_state]
    )

    #yes_btn.click(fn=on_click, inputs=["yes", index_state], outputs=[image_display, button_row, info_text, index_state])
    #no_btn.click(fn=on_click, inputs=["no", index_state], outputs=[image_display, button_row, info_text, index_state])
    done_btn.click(fn=export_csv, outputs=info_text)

    demo.load(fn=get_next, inputs=index_state, outputs=[image_display, button_row, info_text])

demo.launch(share=True)
