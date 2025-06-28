import os
import csv
from PIL import Image, ImageDraw
import tempfile

def label_id_to_name(label_id):
    return f"class{label_id}"

def review_multi_annotations(images_dir, labels_dir, output_csv):
    image_files = sorted([
        f for f in os.listdir(images_dir) if f.endswith('.jpg')
    ])

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'width', 'height'])

        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))

            if not os.path.exists(label_path):
                print(f"⚠️ Label file not found for {image_file}, skipping.")
                continue

            with Image.open(image_path) as img:
                img_width, img_height = img.size
                draw = ImageDraw.Draw(img)

                bboxes = []
                with open(label_path, 'r') as f:
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

                        # Draw
                        x0, y0 = bbox_x, bbox_y
                        x1, y1 = bbox_x + bbox_width, bbox_y + bbox_height
                        draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
                        draw.text((x0, y0), label_name, fill='red')

                        bboxes.append([
                            label_name,
                            round(bbox_x, 2),
                            round(bbox_y, 2),
                            round(bbox_width, 2),
                            round(bbox_height, 2),
                            image_file,
                            img_width,
                            img_height
                        ])

                # Show image
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                    img.save(temp.name)
                    os.system(f'xdg-open "{temp.name}"')  # for Linux; use 'open' on macOS, 'start' on Windows

                user_input = input("✅ Accept this image and labels? [y/n]: ").strip().lower()
                if user_input == 'y':
                    for bbox in bboxes:
                        writer.writerow(bbox)

                os.unlink(temp.name)  # delete the temp file

    print("✅ Review complete. Accepted entries saved to:", output_csv)

if __name__ == "__main__":
    # # Example usage:
    # images_dir = "/path/to/images"
    # labels_dir = "/path/to/labels"
    # output_csv = "/path/to/output.csv"
    review_multi_annotations(
        images_dir="/home/kaelan/Projects/Github/Grounding-Dino-FineTuning/water/ProcessedDataSet/multi_annot_images",
        labels_dir="/home/kaelan/Projects/Github/Grounding-Dino-FineTuning/water/ProcessedDataSet/multi_annot_labels",
        output_csv="/home/kaelan/Projects/Github/Grounding-Dino-FineTuning/water/ProcessedDataSet/accepted_multiMarked_images.csv"
    )