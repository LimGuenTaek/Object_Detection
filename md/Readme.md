# Object Detection in 20 Years : A Survey

Paper : https://arxiv.org/pdf/1905.05055.pdf
---

### Organization of the Paper

 #### section 1. introduction
 
 #### [section 2. object detection in 20years](https://github.com/LimGuenTaek/Object-Detection/blob/main/Object%20Detection%20in%2020years.md)
 
 #### section 3. Speed Up Detection 
 
 #### section 4. Recent Advances in Object Detection 
 
 #### section 5. Applications 
 
 #### section 6. Conclusions and future directions 
 
---
 
## Introduction

- 객체 감지(object detection)은 디지털 이미지에서 특정 클래스의 시각적 객체(예 : 사람 , 동물 , 자동차)를 감지하는 중요한 컴퓨터 비전 작업이다.

- 객체 감지(object detection)의 목적은 컴퓨터 비전 응용 프로그램에 필요한 가장 기본적인 정보 중 하나를 제공하는 컴퓨터 모델과 기술을 개발하는 것이다 : what objects are where?

- 컴퓨터 비전의 근본적인 문제 중 하나로서, 객체 감지(object detection)는 많은 다른 컴퓨터 비전 과제의 기초를 형성한다.
	* instance segmentation
	* image captioning
	* object tracking

- 응용 프로그램의 관점에서 객체 감지를 두 개의 연구 주제로 그룹화할 수 있습니다.

  * **general object detection** : aims to explore the methods of detecting different types of objects under a unified framewoork to simulate the human vision and cognition

  * **detection application** : refers to the detection under specific application scenarios , such as pedestrian detection , face detection , text detection

- object detection은  autonomous driving , robot vision , video surveillance 등 이미 실생활 에서도 만나볼 수 있다.

- 본 paper가 기존의 review와는 다른 점

	**1. 기술적 진화를 고려한 포괄적인 review**
	
		- 대부분의 이전 검토는 전체적인 기간에 대한 기술적 진화를 고려하지 않고 짧은 과거 기간이나 일부 object detection 작업에만 초점을 맞췄다.
		
		- 역사의 고속도로에 서는 것은 독자들이 완전한 지식 계층 구조를 형성하도록 도울 뿐만 아니라 이 빠르게 발전하는 분야의 미래 방향을 찾는데에도 도움이 된다.
		
	**2. 핵심기술과 최신기술에 대한 심층적인 탐구**
	
		- 발전이 해를 거듭할 수록 많은 technique들이 등장했다 "multiscale detection" , "hard negative mining" , "bounding box regression"
		
		- 하지만 이전의 review들에선 이러한 technique들에 대한 기본적이고 , 기술적인 분석들이 부족한 경우가 많았다.
		
		- 본 paper에서는 이러한 문제들에 대한 깊은 이해를 할 수 있을 것이다.
		
	**3. 감지 속도 향상 기법에 대한 종합적인 분석**
	
		- 본 paper에서는 지난 20년간 object detection의 속도 향상을 "detection pipeline" , "detection backbone" , "numerical computation" 등 다양한 단계로 설명할 것이다.
		
		- 이러한 주제는 이전 review에서 거의 다루지 않았다.

---
