function result=ISNR(x,xx,Y)
%
%  x:  original image
%  Y:  blurred image
%  xx: de-blurred image
%

e1=x-Y;
e2=x-xx;
E1=mean2(e1.*e1);
E2=mean2(e2.*e2);
result=-10*log(E1/E2)/log(10);