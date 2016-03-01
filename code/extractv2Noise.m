
%====================Reading the binary file to get image pixels and labels=========================

img = loadMNISTImages('train-images-idx3-ubyte');
label = loadMNISTLabels('train-labels-idx1-ubyte');
N=60000;

for i=1:N
	img(1:end,i) = add_awgn_noise(img(1:end,i),10);
end
%fclose("all");
%================================= Training the data =================================================

Wji=rand(785,100)*0.09;
Wkj=rand(100,10)*0.09;

theta = 2.5;
DelJ =2.6;
eta=0.6;
Tk = zeros(10,1);

format long

while(DelJ > theta)

	DelJ = 0;
	for input_i = 1:N

		input = [ 1 ; img(:,input_i) ]; % extra input as bias 785 X 1
		hiddenop = Wji' *  input;
		Yj = arrayfun(@sigmoid,hiddenop);

		Zk = Wkj' * Yj;
		Zk = arrayfun(@sigmoid,Zk);

		Tk(1:10)=.02;

		lbl = label(input_i);
		Tk( label(input_i)+1 )=.98;

		Dk = (Tk - Zk) .* (Zk .* ( 1 - Zk )); % Deltak = Tk - Zk * f'(netk)'

		Wkj = Wkj + eta * (Yj * Dk');
		
		Dj = (Yj .* (1 - Yj ) ) .* ( Wkj * Dk ); % f'(netk)' Wkj * Dk
		Wji = Wji + eta * ( input * Dj');
		DelJ = DelJ + sum( (Tk - Zk) .* (Tk - Zk) )*0.5;
	
	end
	display(DelJ);
	
end