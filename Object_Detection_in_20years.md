## Object Detection in 20 years

> object detection을 detectors , evolution of key technique등 시간적 흐름대로 살펴볼 예정

### 1. Road Map of Object Detection

<img width="720" alt="스크린샷 2021-01-29 오후 2 04 57" src="https://user-images.githubusercontent.com/70448161/106234293-e89b0c80-623b-11eb-99a7-cc24ef0e9dbf.png">

#### Traditional Detectors , Before CNN

- 2012년 CNN이 등장하기 전의 object detection task들은 handcrafted feature을 기반으로 설계가 이루어졌다.

- 효과적인 이미지 표현/처리 방법의 부족으로 직접 feature을 고안해내야만 했다.


	**1.Viola Jones Detectors** 
	
		* 얼굴 감지 문제에 좋은 성능을 보임 , 거의 표준과도 같은 방식
		
		* straight foward detection 방식을 사용했음 , 사람 얼굴에서 일반적으로 나타나는 명암을 특징으로 활용한다.
		
		* "integral image" , "Feature selection" , "Detection cascades" 등의 technique을 사용해 speed를 많이 올림
		
	**2.HOG Detectors(Histogram of Oriented Gradients)**
	
		* 일반적으로 보행자 검출이나 사람의 형태에 대한 검출에 많이 사용된다
	
		* HOG는 당시 sclae-invariant feature transform 및 shape context의 중요한 개선으로 간주되었다.	
		
		* Histogram of Oriented Gradients의 줄임말로 image의 지역적 gradient를 해당영상의 특징으로 사용하는 방법
	
	**3.DPM(Deformable Part-based Model)**
	
		* 전통적인 object detector의 최고봉 , divide and conquer 방식을 사용 
		
		* 예를 들어 자동차를 검출할 땐 자동차의 창문 , 자동차의 몸체 , 자동차의 바퀴 등 작은 부분을 검출하는 문제로 고려될 수 있기 때문 
		
		* DPM은 현재 mixture 모델 , hard negative mining , boundingbox regression 문제에 영향을 많이 줬음
		
		
#### CNN based Two stage Detectors

- handcrafted 의 성능이 포화상태가 되고 object detection 분야는 나름의 안정기에 접어듬

- 2012년 세상은 convolution neural network의 탄생을 목격하게 된다.

- deep convolutional network가 robust하고 high level인 feature들을 학습할 수 있게 되자 연구자들은 이를 object detection에 사용하기 시작함

- deep learning 시대에 object detection은 두가지로 나뉘게 된다 "One-stage Detector" , "Two-Stage Detector"


	**1.RCNN** 
	
		* object proposal들을 추출함으로써 시작한다.
		
		* 그리고 나선 각각의 proposal들이 rescale되면서 CNN 모델에 feature들을 얻기 위해 투입된다.
		
		* 마지막으로 linear support vector machine classifier을 사용하여 분류작업을 진행한다.
		
		* VOC07에서 상당한 성능이 증가했지만 detection 속도가 현저하게 느리다는 한계점이 존재했다. 같은해 SPPNet이 이러한 문제를 극복하기 위해 등장했다.
		
	**2.SPPNet**
	
		* SPPNet의 주요 업적은 Spatial Pyramid Pooling(SPP layer)의 도입이라고 할 수 있다.
		
		* 기존의 CNN 아키텍쳐들은 모두 입력 이미지가 고정되어야 했는데 (ex. 224 x 224)  하지만 이렇게 되면 물체의 일부분이 잘리거나, 본래의 생김새와 달라지는 문제점이 있었음
		
		* SPP layer는 input image의 크기에 상관없이 CNN을 적용할 수 있도록 만든 기법
		
		* RCNN에 비해 speed는 20배 정도 빨라졌지만 여전히 training 이 multi-stage인점 , fine tuning 시에는 SPP를 거치기 이전의 layer들은 학습 시키지 못한다는 점 등의 한계가 존재한다
	
	**3.Fast RCNN**
	
		* Fast RCNN의 주요 업적은 CNN fine tuning bounding box regression , classification을 모두 하나의 네트워크에서 학습 시키는 end to end 기법을 제시하였음
		
		* 비록 Fast-RCNN이 R-CNN 과 SPPNet의 장점을 잘 통합했지만 여전히 speed는 한계가 있었음
		
		
	**4.Faster RCNN**
	
		* 첫번째로 Real Time Object Detection의 시작을 알린 기법
		
		* Region Proposal 단계를 Neural Network안으로 끌어와서 진정한 End to End 모델을 제시함
		
		* 모든 단계를 다 합쳐서 5fps 속도를 내며 Pascal VOC기준으로 78.8%의 성능을 냄
		
	**5.Feature Pyramid Networks**
	
		* Object Detection 분야에서 풀리지 않았던 고질적인 난제는 바로 작은 물체를 탐지해내기가 어렵다는 점이었음
		
		* 이를 위해서 이미지나 피처맵의 크기를 다양한 형태로 rescale하는 접근 방식이 있었음
		
		* Feature Pyramid Network의 핵심 idea는 먼저 신경망을 통과하면서 단계별로 feature map들을 생성한 뒤 , 가장 상위 layer에서 거꾸로 내려오면서 feature를 합쳐준 뒤 , Object Detection을 진행함 
		
		* 이러한 방식을 통해서 상위 레이어의 추상화 된 정보와 하위 레이어의 작은 물체들에 대한 정보를 동시에 살리면서 Detection을 수행할 수 있게 됨
		
		* FPN은 이후 등장하는 모델에 큰 영향을 주었으며 그 만큼 Object Detection 분야에서 영향력이 큰 논문임 
		

#### CNN based One stage Detectors

- Two stage detector들의 본질적인 문제인 speed 문제를 해결하기 위해 , "Localization" 과 "Classfication" 문제를 한꺼번에 처리 하는 One-stage Detector들이 등장하기 시작한다.


	**1.YOLO** 
	
		* real time object detection에 혁명을 몰고 온 detector
		
		* 기존에 region proposal 그리고 classification 이렇게 두 단계로 나누어서 진행하던 방식에서 proposal 단계를 제거하고 한번에 Detection을 수행하는 구조
		
		* 하지만 YOLO는 localization에서 그 중에서도 특히 작은 object들에 대해서는 정확성이 떨어지는 한계가 있다.
		
	**2.SSD**
	
		* SSD의 핵심은 multi-reference and multi-resolution detection 기법의 도입이라고 볼 수 있다.
		
		* Multi-feature map 기법의 사용으로 one-stage 기법에서도 정확성을 꽤 향상시킬 수 있었음
		
		* 기존의 여러 리서치 결과들을 잘 통합하여 정확도와 속도를 모두 잡아낸 모델을 만들었음
	
	**3.RetinaNet**
	
		* 높은 속도와 단순성에도 불구하고 one-stage Detector는 two-stage Detector에 비해서 정확도가 낮을 수 밖에 없었음
		
		* RetinaNet에서는 그 이유가 극단적인 클래스 불균형 문제라고 설명한다. 이 문제를 해결하기 위해서 Cross Entropy loss함수를 조금 수정한 Focal Loss를 제안한다. 
		
		* Focal Loss 는 잘 분류된 예제들에 대해서는 작은 가중치를 부여하는 반면 분류하기 어려운 일부 예제들에는 큰 가중치를 부여해서 학습을 어려운 예제에 집중시킨다.
		
		* 그래서 쉽게 분류되는 대부분의 negative 샘플들에 의해서 학습이 압도 되는 문제를 해결 할 수 있음
		
		* Focal Loss는 one-stage detector를 two-stage detector와 비교할만한 정확도를 가능하게 했다.
	
### 2. Technical Evolution in Object Detection

#### Technical Evolution of Multi-Scale Detection

	- object를 다른 크기와 다른 비율로 검출하는 Multi-Scale detection 기법은 object detection에 있어서 주된 기법이었다.

	- 시기 별로 key technique이 된 Mutl-scale Detection을 소개하겠다.

* **Feature pyramids + sliding windows (before 2014)**

    * Early detection models , VJ , HOG deatector들은 고정된 aspect ratio로 object detection을 수행했는데 고정된 aspect ratio는 complex한 problem을 제대로 처리 하지 못함

    * “mixture model”이란것이 등장하여 그 당시에는 best solution이였다. 

    * 덕분에 different한 aspect ratio들로 detection을 수행하였고 성능이 개선됨을 확인 할 수 있었다.

    * 하지만 object detection task들이 점점 더 복잡해지면서 하나의 통합된 multi-scale 기법이 없을 까 라는 의문이 떠올랐고

    * 이는 object proprosal의 등장 배경이 된다.

* **Dectection with object proposals(2010-2015)**

    * object proposal은 object에 대한 후보군을 나타내는 것이라고 생각하면 된다.

    * object proposal 기법은 방대한 양의 연산을 주여주는 효과를 일으켰다.

    * 초기의 proposal detection은 bottom-up detection 방식을 진행했고 특히 시각적으로 두드러진 object에 민감했다.

    * 2014년 이후 CNN의 인기와 더불어 object proposal detection은 점차 top-down방식으로 바뀌었다.

    * 그러다 연구자들은 다음과 같은 질문에 도달했다 object proposal 기법은 정확성을 위한 기법인가 속도향상을 위한 기법인가?

    * 이 다음에 등장하는 deep regression이 이 질문에 답을 해줄 것 이다.


* **Deep regression(2013-2016)**

    * GPU의 컴퓨팅 능력이 좋아지면서 사람들은 점점 multi-scale detection 문제를 무차별 대입 방식으로 다루게 되었다.

    * deep regression을 이용한 multi-scale detection idea는 굉장히 단순한데 deep learning 기반으로 얻어진 feature들을 바탕으로 바로 bounding box들의 위치를 예측하는 것이었다.

    * 이러한 방법의 장점은 쉽고 단순한데 단점은 너무 단순하기 때문에 localization 정확도가  떨어진다는 점이었다.

    * 이러한 문제를 해결하기 위해 Multi-reference 기법이 등장하게 된다.


* **Multi-references / resolution detection(after 2015)**

    * Multi-reference detection은 multi-scale object detection 에서 가장 많이 사용되는 framework이다.

    * main idea는 기준이 될 box들을 사전에 다른 크기와 다른 비율로 제각각 다른 위치에 위치시킨뒤 prediction을 진행하는 것

    * category 분류를 위해서 cross entropy를 사용하고 localization 예측을 위해 L1/L2 regression loss를 사용한다.

    * 또 다른 중요하고 인기있는 기법은 multi resolution detection인데 layer를 여러개를 사용한다는 점에서 차별점이 있다.

    * CNN의 구조가 피라미드 모양의 Featuremap을 형성하는 구조 이기에 큰 물체는 깊은 layer에서 탐지하기가 용이하고 작은 물체는 얕은 layer에서 탐지가 용이하다는 점을 이용한 것


	
<img width="600" alt="스크린샷 2021-02-01 오후 3 49 06" src="https://user-images.githubusercontent.com/70448161/106424324-5433e280-64a5-11eb-92b5-50159a825a19.png"> 


#### Technical Evolution of BoundingBox Regression

	- Bounding Box regression 은 object detection에 중요한 기술이다.

	- initial proposal 또는 anchor box를 기반으로 예측된 bounding box의 위치를 세분화하는 것을 목표로 합니다.

* **Without BB regression (before 2008)**

    * CNN등장 이전에 진행되었던 object detector들은 Bounding box regression을 사용하지 않았다.

    * sliding window를 통해 바로 detection result를 도출했다.

    * BB regression 기법이 아직 존재하지 않았기 때문에 그저 좀더 촘촘한 feature pyramid를 만들고 좀더 조밀하게 window를 움직일 수 밖에 없었다.

* **From BB to BB (2008-2013)**

    * Bounding box Regression이 처음 등장한것은 DPM 이었다.

    * 그 당시 BB regression은 후처리 과정으로 필수적인 과정은 아니였다.

    * PASCAL VOC의 목표는 각 object에 대한 단일 bounding box를 예측하는 것이므로 DPM이 최종 detection을 만들어 내는 가장 간단한 방법은 root filter의 위치를 직접 사용하는 것

    * 이후 , R Girshick 등은 object hypothesis를 완벽하게 구성하여 경계 상자를 예측하고 이 과정을 선형 최소 제곱 회귀문제로 공식화하는 보다 더 복잡한 방법을 도입했다.


* **From features to BB (after 2013)**

    * Faster RCNN의 도입으로 BB regression은 더이상 후처리 과정이 아닌 detector와 통합된 모습으로 나타났다.

    * 동시에 BB regression은 CNN feature들을 바탕으로 더 발전하고 있었다.

    * 좀 더 robust한 prediction을 얻기위해 smooth L1 loss 나 root-square loss를 사용했다.

    * 또 몇몇의 연구자들은 normalization을 통해 좀 더 robust한 result를 얻기도 했다.
    
<img width="600" alt="스크린샷 2021-02-02 오후 5 33 45" src="https://user-images.githubusercontent.com/70448161/106573361-de9b4580-657c-11eb-8589-bb45bc5951d5.png">

#### Technical Evolution of Non-Maximum Suppression

	- 비 최대치 억제는 object detection에서 중요한 기술 그룹

	- 인접한 창은 대개 유사한 detection score를 가지므로, 비 최대치 억제는 복제된 경계 상자를 제거하고 최종 결과를 얻기 위한 사후 처리 단계로 사용된다.

* **Greedy selection**

    * Greedy selection 방식은 오래된 방법이지만 또한 가장 많이쓰이는 방법이기도 하다.

    * 하나에 ground truth에 대해 class score 가 가장 높은 bounding box만 살려주고 다른 box들은 제거하는 방식

    * 가장 많이 사용되는 방법이지만 여전히 개선의 여지는 있다.

    * 가장 높은 score라고 해서 best fit box는 아니다. section 4-4에서 개선점이 잘 나와있다.


* **BB aggregation**

    * 여러겹의 bounding box들을 combining , clustering 해서 하나의 최종 bounding box를 만들어 내는 것이 idea

    * 이러한 방법의 장점은 객체와 박스들의 관계와 공간 배치를 충분히 고려한다는 것


* **Learning to NMS**

    * NMS를 end to end 방식으로 네트워크의 일부로 훈련시키는 filter로 생각하는 것

    * 이러한 방법은 기존의 직접 수행하는 NMS방법 보다 occlusion 과 조밀한 object detection 을 수행하는데 유망한 결과를 보여주었음

<img width="600" alt="스크린샷 2021-02-02 오후 5 33 52" src="https://user-images.githubusercontent.com/70448161/106573391-e9ee7100-657c-11eb-9fa2-1098f2e2d3c1.png">

#### Technical Evolution of Hard Negative Mining

	- Object detector의 훈련은 필연적으로 불균형 데이터 학습 문제이다.

	- 모든 background 데이터를 사용하는 것은 막대한 양의 negative data들이 학습 과정을 압도할 가능성이 있으므로 위험하다.

* **Bootstrap**

    * 원래 초기에는 단순히 연산량을 줄이기 위해 고안 됐는데 나중에는 DPM , HOG detector에서 불균형 문제를 해결하기 위해 사용됨

    * 전체 데이터 셋을 트레이닝 하는 것은 시간도 오래걸리고 불균형 문제로 인해 학습이 제대로 되지 않을 것 이다.

    * 이럴 때 bootrstrapping 을 이용하여 각 iteration에 사용되는 학습 데이터의 갯수를 줄이고 , 어려운 샘플에 큰 비중을 둬서 학습 하도록 조절 할 수 있다.
    
	[Bootstrap]
	
        * 전체 training sample 중에서 n개를 추출하여 모델을 학습 시킨다.
        * 학습된 모델을 이용하여 training sample을 분류(classify)한다.
        * 잘못 분류된 학습 데이터가 선택될 probability 를 높이고 제대로 분류된 데이터가 선택될 Probability를 낮춘다.
        * 이렇게 하면 다음 번 학습 시 “어려운” 샘플의 비율이 커진다. 
        * 이런 과정을 반복한다.


* **Hard Negative Mining in deep learning based detectors**

    * deep learning 시대가 도래하고 컴퓨팅 파워의 향상으로 bootstrap은 object detection에서 버려졌다.

    * 데이터 불균형 문제를 지우기 위해 faster RCNN이나 YOLO같은 경우는 단순히 positive와 negative window의 가중치들을 조절했다.

    * 하지만 이런 가중치 밸런싱이 본질적으로 불균형 문제를 해결할 수 없다는 것을 알고 다시 bootstrap이 deep learning base인 detector사이에서 재 도입 됐다.
    
<img width="600" alt="스크린샷 2021-02-02 오후 5 34 02" src="https://user-images.githubusercontent.com/70448161/106573408-ed81f800-657c-11eb-803f-12379a9b57bc.png">

---
