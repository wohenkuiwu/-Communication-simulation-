clc;
clear;

%% 这里QPSK调制的思路使用正交调制的思路
%s(t) = A/sqrt(2)[I(t)*cos-Q(t)*sin]
%滤波时 filter和conv的结果不一样

%% 参数初始化
bit_rate = 1e3;    %这里的比特速率是传输多少个0或1
symbol_rate = 500;  %单位码元所携带的比特信息 2bsp中比特率等于符号率 QPSK就等于两倍的BPSK
fc = 2000;
% SNR = 100;
rollfactor = 0.5; %滚降滤波器滚降系数
sps = 16;    %数字采样的采样点数
fs = sps*symbol_rate;   %归一化频率 = 归一化的频率是数字频率，最高频对应pi，对应采样率的fs/2 
bit_num = 1000;
%% 数据转换
data = randi([0 1],1,bit_num);
idata = data(1:2:end);  %同向分量的二进制数字信息
qdata = data(2:2:end); %正交分量的二进制数字信息

%将数字信息转换为双极性不归零码（信源编码）
idata_code = 2 * idata -1;
qdata_code = 2 * qdata -1;

%将双极性不归零吗映射到星座图中pi/4
for j = 1:length(idata)
    if idata_code(j)==1 && qdata_code(j)==1
        idata_code(j) = 1/sqrt(2);
        qdata_code(j) = 1/sqrt(2);
    elseif idata_code(j)==1 && qdata_code(j)== -1
        idata_code(j) = 1/sqrt(2);
        qdata_code(j) = -1/sqrt(2);
    elseif idata_code(j)==-1 && qdata_code(j)== 1
         idata_code(j) = -1/sqrt(2);
        qdata_code(j) = 1/sqrt(2);
    elseif idata_code(j)==-1 && qdata_code(j)== -1
        idata_code(j) = -1/sqrt(2);
        qdata_code(j) = -1/sqrt(2);
    end
end

%% 发送滤波器（信道信号形成器）
rcos_fir = rcosdesign(rollfactor,6,sps,'sqrt');

%这里的插0操作是为了补齐采样点数
idata_upsample = upsample(idata,16);
qdata_upsample = upsample(qdata,16);

% fvtool(rcos_fir,'Analysis','impulse');
% freqz(rcos_fir);

%经过根升余弦滤波器后的信号值
%问题：经过有限长滤波器后的时延问题？
%解决：有限长滤波器的构成多阶数级联形成，
% fir_idata_upsample = filter(rcos_fir,1,idata_upsample);
% fir_qdata_upsample = filter(rcos_fir,1,qdata_upsample);

fir_idata_upsample = conv(idata_upsample,rcos_fir,'same');
fir_qdata_upsample = conv(qdata_upsample,rcos_fir,'same');


%% 数字带通调制（分为同向和正交）
time = [1:length(fir_idata_upsample)]; %有多少个码元，就有多少个载波

%将同向和正交分量合并为QPSK信号
rcos_full_data = fir_idata_upsample.*cos(2*pi*fc.*time/fs)-fir_qdata_upsample.*sin(2*pi*fc.*time/fs);
% fvtool(rcos_idata_upsample,'Analysis','impulse');
% freqz(rcos_full_data);

%% 带通调制信号经过信道（加入高斯白噪声）
ebn0 = [-50:50];
snr = ebn0 + 10*log10(2) - 10*log10(0.5*16);

for i = 1:length(snr)
    
    %对复数信号接入高斯白噪声
    rcos_full_data_noise = awgn(rcos_full_data,snr(i),'measured');
    
    %使用相干解调接收
    idata_mes_noise = rcos_full_data_noise .* cos(2*pi*fc.*time/fs);
    qdata_mes_noise = rcos_full_data_noise .* -sin(2*pi*fc.*time/fs);
    
    %匹配滤波器同样滤除了高频分量
%     rcos_fir = rcosdesign(rollfactor,6,sps);
%     idata_mes = filter(rcos_fir,1,idata_mes_noise);
%     qdata_mes = filter(rcos_fir,1,qdata_mes_noise);

    idata_mes = conv(idata_mes_noise,rcos_fir,'same');
    qdata_mes = conv(qdata_mes_noise,rcos_fir,'same');

%     freqz(idata_mes)
    
    %接收和发送端使用的跟升余弦滤波器导致延时
%     delay_point = 96*2/2;
    rcos_idata_sample = idata_mes(1:sps:end);
    rcos_qdata_sample = qdata_mes(1:sps:end);
    
    %将数值还原为双极性不归零码
    %这里转换有问题，为什么转换成了0
    rcos_idata_final = sign(rcos_idata_sample);
    rcos_qdata_final = sign(rcos_qdata_sample);
    
    %将接收的-+1转为星座点上的0或1数值
    for j = 1:length(rcos_idata_final)
        if rcos_idata_final(j)==1 && rcos_qdata_final(j)==1
            rcos_idata_final(j) = 1;
            rcos_qdata_final(j) = 1;
        elseif rcos_idata_final(j)==1 && rcos_qdata_final(j)== -1
            rcos_idata_final(j) = 1;
            rcos_qdata_final(j) = 0;
        elseif rcos_idata_final(j)==-1 && rcos_qdata_final(j)== 1
            rcos_idata_final(j) = 0;
            rcos_qdata_final(j) = 1;
        elseif rcos_idata_final(j)==-1 && rcos_qdata_final(j)== -1
            rcos_idata_final(j) = 0;
            rcos_qdata_final(j) = 0;
        end
    end
    
    %误码率检验
    reciver_data = zeros(1,bit_num);
    reciver_data(1:2:end) = rcos_idata_final;
    reciver_data(2:2:end) = rcos_qdata_final;
    [number(i) ,ratio_err(i)] = biterr(reciver_data,data);
end

%% 误码率画图
figure(2)
ber = berawgn(ebn0,'psk',4,'nondiff');
% semilogy(ebn0,ratio_err,'*',ebn0,ber,'+');
semilogy(ebn0,ratio_err,'*');
xlabel('比特信噪比');
ylabel('不同信噪比的误码率');
legend('实验曲线','理论曲线');
grid on
