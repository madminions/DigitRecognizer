
%====================Reading the binary file to get image pixels and labels=========================

trainImg= fopen('train-images-idx3-ubyte','r','b');
N=60000;
fseek(trainImg,16,'bof');

img2 = fread(trainImg,28*28,'uchar');% each image has 28*28 pixels in unsigned byte format
img = img2;

for i=1:N
	img2 = fread(trainImg,28*28,'uchar');% each image has 28*28 pixels in unsigned byte format
	img = [ img img2 ];
end

trainlbl = fopen('train-labels-idx1-ubyte','r','b'); % first we have to open the binary file
fseek(trainlbl,8,'bof');

label2 = fread(trainlbl,1,'uchar');
label=label2;

for i=1:N
	label2 = fread(trainlbl,1,'uchar');
	label = [ label label2 ];
end
fclose("all");

ConfusionMatrix = knn(img,label)
noOfTestImages = 10000;
	sum=0;
	for i=1:10
		sum+=ConfusionMatrix(i,i);
	end

	printf("Accuracy is %f\n",sum/100);
	ConfusionMatrix

	for i=1:10
		Recall = ConfusionMatrix(i,i)/(sumMatrix(ConfusionMatrix(i,:)') );
		printf("Recall for  %d is %f\n",i,Recall);
	end

	for i=1:10
		specificity = (noOfTestImages-sumMatrix(ConfusionMatrix(i,:)')...
			-sumMatrix(ConfusionMatrix(:,i)) + ConfusionMatrix(i,i))...
			/(noOfTestImages - sumMatrix(ConfusionMatrix(i,:)') ) ;
		printf("Specificity for  %d is %f\n",i,specificity);
	end


	for i=1:10
		precision = ConfusionMatrix(i,i)/(sumMatrix(ConfusionMatrix(:,i)) );
		printf("Precision for  %d is %f\n",i,precision);
	end

printf("Accuracy is %f ",trace(ConfusionMatrix)/100 );