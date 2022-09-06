%% 初始化参数设置
T = 1;                %采样时间为1s
bit_rate = 1e3;       %这里的比特速率是1s传输多少个0或1
symbol_rate = 1e3;    %单位码元所携带的比特信息 2bsp中比特率等于符号率 QPSK就等于两倍的BPSK
sps = 16;             % 由模拟信号转换为数字信号时的采样点数
fc = 2e3; 
fs = sps * symbol_rate;    %代表了采样频率（单位为赫兹），即采样出1s(一个周期)内完整的一个波形
rollof_factor = 0.5;    %滚降因子


%% 对输入的二进制01信息进行处理，转换成双极性不归零码
%[]方括号组成的信源信息,这里的信息指的是机器中的0或者1信息
message_source = [ones(1,20) zeros(1,20) randi([0 1],1,99960)];

%将比特信息映射为双极性不归零信息 A -A ==>转换为符号信息
bpsk_message = 2 * message_source -1 ;


%% 设置信道信号形成器，得到滤波器的冲激响应(使用根升余弦滤波器进行发送成型)
rcos_fir = rcosdesign(rollof_factor,6,sps,'sqrt'); % 这里的6和sps控制通过滤波器的信号长度
%根升余弦滤波器被截断在span个符号位，每个符号都包含sps个采样点数

% fvtool(rcos_fir,'Analysis','impulse')  %表示时域信号图
% freqz(rcos_fir)     %表示频率图

%进行插值，每个符号需要采样16个点，凑齐需要的采样点数
% for i=1:length(bpsk_message)
%     up16_bpsk_message(1+16*(i-1)) = bpsk_message(i) ;
%     up16_bpsk_message(2+16*(i-1):16*i) = zeros(1,15);
% end
up16_bpsk_message = upsample(bpsk_message,16);



%% 将形成的信号通过滤波器
rcos_bpsk_message = filter(rcos_fir,1,up16_bpsk_message);
% rcos_bpsk_message = filter(rcos_fir,1,up16_bpsk_message);

%波形观察
% figure(1);
% plot(rcos_bpsk_message);
% title('时域波形');
% figure(2);
% plot(abs(fft(rcos_bpsk_message)));
% title('频域波形');

%% 将信号进行调制 s(t)*cos(2*pi*fc*n/fs)
time = [1:length(rcos_bpsk_message)];
bpsk_carrier = rcos_bpsk_message.*cos(2*pi*fc.*time/fs);

% 波形观察
% figure(1);
% plot(bpsk_carrier);
% title('时域波形');
% figure(2);
% plot(abs(fft(bpsk_carrier)));
% title('频域波形');

%% 模拟信号通过信道
%设置信噪比，单位为dB
%help awgn 符号能量与信噪比的换算公式
ebno = [-6:8];
snr = ebno - 10*log10(0.5*16);


%进行了15个点的调试，分别以不同大小的信噪比来检验误码率大小
for i = 1:length(snr)
    %模拟载波通过信道，以snr的方式加入高斯白噪声
    rcos_bpsk_carrier_noise = awgn(bpsk_carrier,snr(i),'measured');
    
    %接收端使用相干解调接受载波
    rcos_mes_carrier_noise = rcos_bpsk_carrier_noise.*cos(2*pi*fc.*time/fs);
    
    %接收端使用相干载波解调时，cos^2会产生高频分量，需要使用低通滤波器截断高频分量
%     fir_low = fir1(128,0.2); %截止频率0.2*（fs/2）
%     %freqz(fir_low);
%     rcos_mes_carrier_noise_low = filter(fir_low,1,rcos_mes_carrier_noise);
    
    %接受端在采用同样的接收匹配滤波器进行接收
    rcos_fir = rcosdesign(rollof_factor,6,sps); % 这里的6和sps控制通过滤波器的信号长度
    rcos_mes_bpsk = filter(rcos_fir,1,rcos_mes_carrier_noise);
    
    %fir滤波器会产生延迟
    decision_point = (96+96)/2;
    
    %每个符号采样一个点进行判决,这里产生了时延，导致抽样点数减少
     rcos_dcision = rcos_mes_bpsk(decision_point:sps:end);
     rcos_final = sign(rcos_dcision);
    
%     figure(1);
%     plot(rcos_final,'-*');
%     title('判决结果');
%     
%     eyediagram(message_source,sps);
%     title('发射端二进制眼图');
%     eyediagram(rcos_final,sps);
%     title('接收端二进制眼图');
    
    %误码率性能对比
    [err_number(i),bit_err_ratio(i)] = biterr(message_source(1:length(rcos_dcision)),(rcos_final+1)/2);
end

%% 仿真结果
ber = berawgn(ebno,'psk',2,'nondiff');
semilogy(ebno,bit_err_ratio,'*',ebno,ber,'+');
xlabel('比特信噪比');
ylabel('不同信噪比的误码率');
legend('实验曲线','理论曲线');
grid on