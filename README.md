# About Usage
1) You should have 2 directory as '1' and '0'.
2) The directory '1' should have photos of your face.
3) The directory '0' should have photos of other people's faces
4) Get your main path which includes the directories '1','0'.
5) Create as classifier = HolyClassifier(main_path)
6) start process as classifier.start()
7) Get prediction with image as pred = classifier.model.predict_with_image(path_of_image) 
