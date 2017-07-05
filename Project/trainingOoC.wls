#!/usr/bin/env wolframscript
(* ::Package:: *)

classesPath = FileNameJoin[{Directory[],"classes.m"}];


$classes = Import[classesPath];


trainingSetIndexPath= FileNameJoin[{Directory[],"trainingSetIndex.mx"}];


validationSetIndexPath= FileNameJoin[{Directory[],"validationSetIndex.mx"}];


netPath = FileNameJoin[{Directory[],"net.wlnet"}];		


checkpointsPath= FileNameJoin[{Directory[],"Checkpoint"}]; 	


net = Import@netPath;


trainingSetIOFiles  = Activate @ Import @ trainingSetIndexPath;


validationSetIOFiles  = Activate @ Import @ validationSetIndexPath;


trainedNet = NetTrain[
	net,
	trainingSetIOFiles,
	ValidationSet -> validationSetIOFiles,
	TrainingProgressCheckpointing -> {
		"Directory", 
		checkpointsPath, 
		"Interval" -> Quantity[1, "Hours"]
	}
	
];


Export[FileNameJoin@{checkpointsPath, "final.wlnet"}, trainedNet];
