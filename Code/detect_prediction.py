from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = '/content/drive/MyDrive/KAIST_Double/tar_bn/93_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
normalize_thermal = transforms.Normalize(mean=[0.485],std=[0.229])


def detect(rgb_image, thermal_image, min_score, max_overlap, top_k, suppress=None):
    # Transform
    rgb = normalize(to_tensor(resize(rgb_image)))
    rgb = rgb.to(device)  
    thermal = normalize_thermal(to_tensor(resize(thermal_image)))
    thermal = thermal.to(device) 

    predicted_locs, predicted_scores = model(rgb.unsqueeze(0),thermal.unsqueeze(0)) # Forward prop.

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [rgb_image.width, rgb_image.height, rgb_image.width, rgb_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return rgb_image

    # Annotate
    annotated_image = rgb_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
        #draw.score 이런식으로 해주고 좌표값 조절만 잘 해주면 될 듯
    del draw
    return annotated_image


if __name__ == '__main__':

    rgb_images=[]
    thermal_images=[]

    f=open("/content/drive/MyDrive/KAIST_Double/test-all-20.txt",'r')
    path=f.readlines()

    for i in range(len(path)):
      if i < 1455:
        rgb_images.append("/content/drive/MyDrive/KAIST_Original/KAIST_dataset/"+path[i][:6]+path[i][:11]+"visible/"+path[i][11:17]+".jpg")
        thermal_images.append("/content/drive/MyDrive/KAIST_Original/KAIST_dataset/"+path[i][:6]+path[i][:11]+"lwir/"+path[i][11:17]+".jpg")
      else :
        rgb_images.append("/content/drive/MyDrive/KAIST_Original/KAIST_dataset/"+path[i][:11]+"visible/"+path[i][11:17]+".jpg")
        thermal_images.append("/content/drive/MyDrive/KAIST_Original/KAIST_dataset/"+path[i][:11]+"lwir/"+path[i][11:17]+".jpg")

    for i in range(len(rgb_images)):
      print(i)
      rgb_image = Image.open(rgb_images[i], mode='r')
      rgb_image = rgb_image.convert('RGB')

      thermal_image = Image.open(thermal_images[i], mode='r')
      thermal_image = thermal_image.convert('L')

      detect(rgb_image, thermal_image, min_score=0.2, max_overlap=0.5, top_k=200).save("/content/drive/MyDrive/KAIST_Double/Prediction/"+str(i)+".jpg",'JPEG')