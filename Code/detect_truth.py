from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(index,truth,original_image, min_score, max_overlap, top_k, suppress=None):
    
    det_boxes=[]
    det_labels=[]
    boxes=[]
    labels=[]

    while True :
      tmp=truth[index]["bbox"]
      tmp[2]=tmp[0]+tmp[2]
      tmp[3]=tmp[1]+tmp[3]
      boxes.append(tmp)
      labels.append(truth[index]["category_id"])
      if(truth[index]["image_id"]==truth[index+1]["image_id"]):
        index=index+1
      else : 
        index=index+1
        break
    
    det_boxes.append(torch.FloatTensor(boxes))
    det_labels.append(torch.LongTensor(labels))
    
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    '''
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    '''
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
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
    del draw

    return annotated_image,index


if __name__ == '__main__':

    images=[]
    idx=0
    cnt=0
    f=open("/content/drive/MyDrive/MyCode/test-all-20.txt",'r')
    path=f.readlines()

    for i in range(len(path)):
      if i < 1455:
        images.append("/content/drive/MyDrive/MyCode/KAIST_dataset/"+path[i][:6]+path[i][:11]+"lwir/"+path[i][11:17]+".jpg")
      else :
        images.append("/content/drive/MyDrive/MyCode/KAIST_dataset/"+path[i][:11]+"lwir/"+path[i][11:17]+".jpg")
        
    with open("/content/drive/MyDrive/MyCode/Json/kaist_annotations_test20.json", 'r') as f: ## ground truth 값 읽어오기
      truth = json.load(f)["annotations"]
    
    while idx<=4238:
      
      original_image = Image.open(images[truth[idx]["image_id"]], mode='r')
      original_image = original_image.convert('RGB')
      image_id=truth[idx]["image_id"]
      print("index : {} and image_id : {}".format(idx,image_id))
      image,idx=detect(idx,truth,original_image, min_score=0.2, max_overlap=0.5, top_k=200)
      image.save("/content/drive/MyDrive/MyCode/GroundTruth/"+str(image_id)+".jpg",'JPEG')
      cnt=cnt+1
    print(cnt)
    