
function ConfusionMatrix = knn(trainImg,trainlabel)
	
%====================Reading the testing binary file to get image pixels and labels=========================
noOfImages = 60000;
noOfTestImages = 10000;

testImg= fopen('t10k-images-idx3-ubyte','r','b');

fseek(testImg,16,'bof');
img2 = fread(testImg,28*28,'uchar');% each image has 28*28 pixels in unsigned byte format
img = img2;

for i=1:noOfTestImages
	img2 = fread(testImg,28*28,'uchar');% each image has 28*28 pixels in unsigned byte format
	img = [ img img2 ];
end
testImg = img;

% for label
testlbl = fopen('t10k-labels-idx1-ubyte','r','b'); % first we have to open the binary file
fseek(testlbl,8,'bof');

label2 = fread(testlbl,1,'uchar');
label=label2;

for i=1:noOfTestImages
	label2 = fread(testlbl,1,'uchar');
	label = [ label label2 ];
end
testlabel = label;
ConfusionMatrix = zeros(10,10);

for i=1:noOfTestImages
	for j=1:noOfImages
		dist(1,j) = sqrt( sum( ( testImg(:,i) - trainImg(:,j) ).^2 ));
	end
	[t,index] = min(dist);
	trl = trainlabel(1,index);
	tsl = testlabel(1,i);
	ConfusionMatrix(tsl+1,trl+1)++;
end

fclose("all");
endfunction