# Pedestrian Detection 

**Model** : [Gated Fusion Double SSD for Multispectral Pedestrian Detection](https://arxiv.org/abs/1903.06999)

**Dataset** : [KAIST Multispectral Pedestrian dataset](https://soonminhwang.github.io/rgbt-ped-detection/)

---
### Accuracy

**Baseline** | missrate : 28.89 , Recall : 0.804

**GFD-SSD** | missrate : 16.06 , Recall : 0.908 



### Model

* [SSD tutorial Code](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)을 기반으로 GFU-SSD model을 구현함

* 핵심은 RGB , Thermal을 두개의 network로 따로따로 Convolution을 태워준뒤 Gate-Fusion이라는 방법으로 Feature들을 합쳐주는 것

<img width="500" alt="스크린샷 2021-03-15 오후 2 13 44" src="https://user-images.githubusercontent.com/70448161/111106715-f7921000-8598-11eb-9959-c86601784753.png">

<img width="450" alt="스크린샷 2021-03-15 오후 2 13 52" src="https://user-images.githubusercontent.com/70448161/111106721-f9f46a00-8598-11eb-84ef-d478f446fd39.png">

* C : concat , R : relu , + : element-wise addition , Fc : RGB-Feature , Ft : Thermal-Feature

### Code

* Fusion을 담당해주는 class이며 , 그 외 Code 수정 부분은 input data를 두장 처리해주는(RGB , Thermal) 과정이 대부분 입니다.


```python
class FusionLayer(nn.Module):

  def __init__(self):
    super(FusionLayer, self).__init__()

    self.conv4_3_3_rgb=nn.Conv2d(1024,512,kernel_size=3,padding=1)
    self.conv4_3_3_thermal=nn.Conv2d(1024,512,kernel_size=3,padding=1)
    self.conv4_3_1_fusion=nn.Conv2d(1024,512,kernel_size=1)

    self.conv7_3_rgb=nn.Conv2d(2048,1024,kernel_size=3,padding=1)
    self.conv7_3_thermal=nn.Conv2d(2048,1024,kernel_size=3,padding=1)
    self.conv7_1_fusion=nn.Conv2d(2048,1024,kernel_size=1)

    self.conv8_2_3_rgb=nn.Conv2d(1024,512,kernel_size=3,padding=1)
    self.conv8_2_3_thermal=nn.Conv2d(1024,512,kernel_size=3,padding=1)
    self.conv8_2_1_fusion=nn.Conv2d(1024,512,kernel_size=1)

    self.conv9_2_3_rgb=nn.Conv2d(512,256,kernel_size=3,padding=1)
    self.conv9_2_3_thermal=nn.Conv2d(512,256,kernel_size=3,padding=1)
    self.conv9_2_1_fusion=nn.Conv2d(512,256,kernel_size=1)

    self.conv10_2_3_rgb=nn.Conv2d(512,256,kernel_size=3,padding=1)
    self.conv10_2_3_thermal=nn.Conv2d(512,256,kernel_size=3,padding=1)
    self.conv10_2_1_fusion=nn.Conv2d(512,256,kernel_size=1)

    self.conv11_2_3_rgb=nn.Conv2d(512,256,kernel_size=3,padding=1)
    self.conv11_2_3_thermal=nn.Conv2d(512,256,kernel_size=3,padding=1)
    self.conv11_2_1_fusion=nn.Conv2d(512,256,kernel_size=1)

    self.init_conv2d()

  def init_conv2d(self):
    for c in self.children():
      if isinstance(c, nn.Conv2d):
        nn.init.xavier_uniform_(c.weight)
        nn.init.constant_(c.bias, 0.)

  def Gated_Fusion_Unit(self, feature_rgb, feature_thermal, conv3_rgb, conv3_thermal, conv1): 

    feature_cat=torch.cat([feature_rgb,feature_thermal],dim=1)
    
    feature_cat_rgb=F.relu((conv3_rgb(feature_cat)))
    feature_cat_thermal=F.relu((conv3_thermal(feature_cat)))

    feature_rgb=feature_rgb+feature_cat_rgb
    feature_thermal=feature_thermal+feature_cat_thermal

    feature_return=torch.cat([feature_rgb,feature_thermal],dim=1)

    feature_return=F.relu(conv1(feature_return))

    return feature_return
  
  def forward(self,feature_rgb,feature_thermal): # feature들은 list로 넘겨줌

    conv4_3_feats=self.Gated_Fusion_Unit(feature_rgb[0],feature_thermal[0],self.conv4_3_3_rgb,self.conv4_3_3_thermal,self.conv4_3_1_fusion)
    conv7_feats=self.Gated_Fusion_Unit(feature_rgb[1],feature_thermal[1],self.conv7_3_rgb,self.conv7_3_thermal,self.conv7_1_fusion)
    conv8_2_feats=self.Gated_Fusion_Unit(feature_rgb[2],feature_thermal[2],self.conv8_2_3_rgb,self.conv8_2_3_thermal,self.conv8_2_1_fusion)
    conv9_2_feats=self.Gated_Fusion_Unit(feature_rgb[3],feature_thermal[3],self.conv9_2_3_rgb,self.conv9_2_3_thermal,self.conv9_2_1_fusion)
    conv10_2_feats=self.Gated_Fusion_Unit(feature_rgb[4],feature_thermal[4],self.conv10_2_3_rgb,self.conv10_2_3_thermal,self.conv10_2_1_fusion)
    conv11_2_feats=self.Gated_Fusion_Unit(feature_rgb[5],feature_thermal[5],self.conv11_2_3_rgb,self.conv11_2_3_thermal,self.conv11_2_1_fusion)

    return conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

```

### Prediction


