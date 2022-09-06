clc;
clear;

M = 16;
aqam = [-3,-1,1,3];
A = repmat(aqam,4,1); %对原数组进行复制成（4，1）
B = flipud(A');    %将原数组进行从上向下的翻转
konst_qam = A+1j*B; %构造出复数形式的振幅星座图表示
konst_qam = reshape(konst_qam,[],1);%产生所有的星座点，对应复数形式
qam = konst_qam(randi([0,15], 10000, 1)+1);%将10000数据信息映射为格雷码的复数信息

%绘制16QAM的星座图
% figure(1)
% plot(qam, 'o');
% title('Ordinary Signal Contellation');
% axis([-4 4 -4 4]);
% xlabel('Re');ylabel('Im')


