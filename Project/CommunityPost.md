**Leaf Classifier**
===============
**Abstract**

**GOAL OF THE PROJECT:** The goal of the project is to classify leaves on the basis of their species using a Convolution Neural Network (CNN) model.

**SUMMARY OF WORK:** A dataset of different species of leaves from Northeastern United States and Canada was taken and was split into training and validation sets. A Convolutional Neural Network (CNN) model was developed for the application of Deep learning in Image Processing. The network was trained on this dataset and a Leaf Classifier was developed.

**RESULTS AND FUTURE  WORK:** The network is able to classify the leaf on the basis of it's species most of the time. With more training and parameters, the accuracy level of the classifier can be improved. This network and be useful on a larger dataset with species of leaves from worldwide and thus has a vast range of possibilities to classify from. With more training and a large variety of datasets a lot can be achieved in the future.

**Main Results in Detail**

The Goal of the project was to create a classifier which correctly classifies the species of the leaf while taking image file as the input and giving the corresponding species of the leaf, the image belongs to. To do so, a Convolutional Neural Network Model was taken into account. The Network Architecture referred to was "SqueezeNet V1.1 Trained on ImageNet Competition Data". The Dataset consisted of 184 different species, thus the network was designed in the way that it gives out the result distributed among 184 different species and one with the highest probability be interpreted as the species of the image.
The network is then trained over a training data set of 9727 images of different leafs with a validation dataset of 2498 images to validate the training of the network. The Accuracy obtained is 91.7%, however is this due to over-fitting during the training. This is mainly due to the less amount of test dataset present.


**Code of the Project :**

Importing the Data set from the local machine and creating a valid dataset as "File-> Class".

    data = SemanticImport["C:\\Users\\haris\\Desktop\\leafDataSet\\leafsnap-dataset-images.txt",<|"image_path"->Automatic,"species"->"String","source"->"String"|>];
    actualData = GroupBy[Normal[data],Last->(Slot["image_path"]->Slot["species"]&)]["field"];
    addFullPath[x_] := FileNameJoin[{"","Users","haris","Desktop","leafDataSet",x}];
    actualData = Normal[KeyMap[File@*addFullPath, Association[actualData]]];
    fileBySpiecies=GroupBy[actualData,Last->First];

Function to create more data as the current data is not enough and storing it in a directory called createdData.

    smallSets=Select[fileBySpiecies,Length[#]<40&]; (* Creating Data for dataset of classes less than 40*)
    SetDirectory["createdData"];
    transformationList={ImageRotate, ImageReflect};
    applyFunction[key_,value_]:=key->Flatten@Table[	
    						Export[StringJoin[ToString[Hash[#]],".jpg"],#]&/@Through[transformationList[img] ],
    						{img,Import/@value}]
    createdData=Map[File@*AbsoluteFileName,KeyValueMap[applyFunction,smallSets],{3}];
    ResetDirectory[];

Merging the created data and the actual data.

    finalData=Flatten[Join[actualData, Map[Reverse,Thread/@createdData,{2}]]];

Separating data into training and validation dataset.

    set1=Keys@GroupBy[Import["C:\\Users\\haris\\OneDrive\\Documents\\finaldata.m"],Values,Keys];
    set2=Values@GroupBy[Import["C:\\Users\\haris\\OneDrive\\Documents\\finaldata.m"],Values,Keys];
    trainingSet=Take[#,Floor[.8*Length[#]]]&/@set2;
    trainingSet=Thread[set1->trainingSet];
    trainingSet=Flatten@Map[Reverse,Thread/@trainingSet,{2}];
    validationSet= Complement[finalData,trainingSet];
    $classes=Union[finaldata[[All,2]]]

Moving All the training Set and the Validation Set to a single Main Directory and Sub Directories as trainingSetData and validationSetData.

    ParallelMap[CopyFile[#, "C:\\Users\\haris\\OneDrive\\Documents\\trainingSetData\\"<>FileNameTake[#]]&, paths];
    trainingFiles=File/@FileNames["C:\\Users\\haris\\OneDrive\\Documents\\trainingSetData\\*"];
    setNames=FileBaseName/@trainingSet[[All,1]];
    fileNames = FileBaseName/@trainingFiles[[All,1]];
    f := Function[
    	Block
    	[
    		{base, pos},
    		
    		base = FileBaseName[#];
    		pos = Position[setNames, base][[1,1]];
    		# -> trainingSet[[pos, 2]]
    	]
    ]
    training=ParallelMap[f,trainingFiles]
    paths=Keys[validationSet][[All,1]];
    ParallelMap[CopyFile[#, "C:\\Users\\haris\\OneDrive\\Documents\\validationSetData\\"<>FileNameTake[#]]&, paths];
    setNames=FileBaseName/@validationSet[[All,1]];
    validationFiles=File/@FileNames["C:\\Users\\haris\\OneDrive\\Documents\\validationSetData\\*"];
    fileNames = FileBaseName/@validationFiles[[All,1]];
    f := Function[
    	Block
    	[
    		{base, pos},
    		
    		base = FileBaseName[#];
    		pos = Position[setNames, base][[1,1]];
    		# -> validationSet[[pos, 2]]
    	]
    ]
    validation=ParallelMap[f,validationFiles]

Creating a Net design.


    squeezeNet=NetModel["SqueezeNet V1.1 Trained on ImageNet Competition Data"];
    
    finalNet=NetChain[Join[Association["imgaug10"->ImageAugmentationLayer[{227,227},"ReflectionProbabilities"-> {0.5,0.5}]],
    Normal@Drop[squeezeNet,-5],
    Association[
    "conv10"->ConvolutionLayer[184,{1,1}],
    "relu_conv10"->Ramp,
    "pool10"->AggregationLayer[Mean],
    "probabilities"->SoftmaxLayer[]
    ]],
    "Input"->NetEncoder[{"Image",{300,300},ColorSpace->"RGB"}],
    "Output"->  NetDecoder[{"Class",$classes}]
    ]

Creating ".MX" and ".WLNET" files for remote training.

    Export["trainingSetIndex.mx",trainingOoC,"MX"]
    Export["validationSetIndex.mx",validation,"MX"]
    Export["net.wlnet",finalNet]

Training of the Network.(Script)

    classesPath = FileNameJoin[{Directory[],"classes.m"}];
    $classes = Import[classesPath];
    trainingSetIndexPath= FileNameJoin[{Directory[],"trainingSetIndex.mx"}];
    validationSetIndexPath= FileNameJoin[{Directory[],"validationSetIndex.mx"}];
    netPath = FileNameJoin[{Directory[],"net.wlnet"}];
    Length@Normal@netPath;
    checkpointsPath= FileNameJoin[{Directory[],"Checkpoint"}]; 	
    netPath;
    net = Import@netPath;
    trainingSetIOFiles  = Activate @ Import @ trainingSetIndexPath;
    validationSetIOFiles  = Activate @ Import @ validationSetIndexPath
    
    (* Round 1 of training *)
    trainedNet = NetTrain[
    	net,
    	trainingSetIOFiles,
    	ValidationSet -> validationSetIOFiles,
    	TrainingProgressCheckpointing -> {
    		"Directory", 
    		checkpointsPath, 
    		"Interval" -> Quantity[1, "Hours"]
    	},
    	LearningRateMultipliers -> {"conv10"->1, _ -> None}
    ]
    
    
    Export[FileNameJoin@{checkpointsPath, "final.wlnet"}, trainedNet];
Round 2 of training.

    netPath = FileNameJoin[{Directory[],"Checkpoint","final.wlnet"}];
    trainedNet = NetTrain[
    	net,
    	trainingSetIOFiles,
    	ValidationSet -> validationSetIOFiles,
    	TrainingProgressCheckpointing -> {
    		"Directory", 
    		checkpointsPath, 
    		"Interval" -> Quantity[1, "Hours"]
    	},
    	LearningRateMultipliers -> {"conv10"->1, _ -> None}
    ]
    
    
    Export[FileNameJoin@{checkpointsPath, "2017-07-05T02_38_18_0_100_12200_5.04e-2_3.85e-1.wlnet"}, trainedNet];

Function to find the accuracy of the network.

    results=net/@validationSetIOFiles[[All,1]];
    
    expected=validationSetIOFiles[[All,2]];
    f[x_,y_]:= N[Count[
    				MapThread[
    						SameQ,
    						{x,y}
    						   ],
    				True]/ Length@validationSetIOFiles];
    f[results,expected]
Data Sources Links/References

"Leafsnap: A Computer Vision System for Automatic Plant Species Identification,"
Neeraj Kumar,Peter N.Belhumeur,Arijit Biswas,David W.Jacobs,W.John Kress,Ida C.Lopez,João V.B.Soares,Proceedings of the 12th European Conference on Computer Vision (ECCV),October 2012

Future Directions

The network to be trained on a big amount of data set and can used for all the species of leaf present on earth. The drawback faced here was the the data set present is really small. In future the network can be trained on big amount of data set with good amount of test set available and thus can be improved exponentially.

Background Info Links/References

Plant Leaf Recognition Using a Convolution Neural Network Wang-Su Jeon1 and Sang-Yong Rhee2 1Department of IT Convergence Engineering, Kyungnam University, Changwon, Korea 2Department of Computer Engineering, Kyungnam University, Changwon, Korea.

**Random Training Data:**

![enter image description here][1]

[GitHub  repository for the project>>][2]

    


  [1]: http://community.wolfram.com//c/portal/getImageAttachment?filename=Capture.PNG&userId=1081661
  [2]: https://github.com/roronoaharish/WSS-17/tree/master/Project
