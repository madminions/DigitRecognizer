
function accuracy(Wkj,Wji)
	
	noOfTestImages = 10000;
	ConfusionMatrix = zeros(10,10);


	testImg = loadMNISTImages('t10k-images-idx3-ubyte');
	label = loadMNISTLabels('t10k-labels-idx1-ubyte');

	for i=1:noOfTestImages
		
		Xi = [ 1 ; testImg(:,i) ];
		
		hiddenop = Wji' *  Xi;
		Yj = arrayfun(@sigmoid,hiddenop);

		Zk = Wkj' * Yj;
		Zk = arrayfun(@sigmoid,Zk);
		%printf("\n");
		Zk;

		[t,index] = max(Zk);
		label(i)+1;
		index;
		ConfusionMatrix(label(i)+1,index)++;
	end
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

endfunction

%{
		1					not 1
1		(i,i)(tp)			row(i)-(i,i)(fn)
not1	col(i)-(i,i)(fp)	all - row(i)-col(i)+(i,i)(tn)

    sensitivity = recall = tp / t = tp / (tp + fn)
    specificity = tn / n = tn / (tn + fp)     = all-row(i)-col(i)+(ii)/all-row(i)
    precision = tp / p = tp / (tp + fp)

%}