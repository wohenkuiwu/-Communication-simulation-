function y_CFO = add_CFO(y,CFO,Nfft)

nn = 0:length(Nfft)-1;
y_CFO = y.*exp(1j*2*pi*CFO*nn/Nfft);