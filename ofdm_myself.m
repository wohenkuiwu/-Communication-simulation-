clear all;
close all;
carrier_count = 200; % 子载波数
symbol_count = 100; %总OFDM符号数，每个符号由200个子载波构成，每个子载波调制一个QAM信息
ifft_length = 512; % IFFT长度
CP_length = 128; % 循环前缀
CS_length = 20; % 循环后缀
rate = [];
SNR =20;
bit_per_symbol = 4; 
alpha = 1.5/32; % 升余弦窗系数

% ================产生随机序列=======================

bit_length = carrier_count*symbol_count*bit_per_symbol;
bit_sequence = round(rand(1,bit_length))'; % 列向量

% =================串并转换==========================
% ==================16QAM调制=========================

% 1-28置零 29-228有效 229-285置零 286-485共轭 486-512置零 
carrier_position = 29:228;
conj_position = 485:-1:286;
bit_moded = qammod(bit_sequence,16,'InputType','bit');

% figure(1);
% scatter(real(bit_moded),imag(bit_moded));
% title('调制后的星座图');
% grid on;

% ===================IFFT===========================
%经过调制后，变成了复数信号
ifft_position = zeros(ifft_length,symbol_count);

%将QAM序列转换为N个并行的符号流
bit_moded = reshape(bit_moded,carrier_count,symbol_count);

figure(2);
stem(abs(bit_moded(:,1)));
grid on;

%512个载波中对应用到的载波进行赋值 a(19:228,:) = b(:,:) 等价于a中19到228全部的列由b来赋值
ifft_position(carrier_position,:)=bit_moded(:,:);
%conj对复数取共轭
ifft_position(conj_position,:)=conj(bit_moded(:,:));

%对ifft_position这个矩阵每一行做ifft_length点的离散傅里叶反变换
signal_time = ifft(ifft_position,ifft_length);

%一个ofdm符号是由IFFT点数构成的==子载波数
figure(3);
subplot(3,1,1)
plot(signal_time(:,1),'b');
title('原始单个OFDM符号');
xlabel('Time');
ylabel('Amplitude');
axis([0 500 -0.5 0.5])


% ==================加循环前缀和后缀==================
%避免由于多径传输的时延所带来的单个OFDM信号内的不同频率的子载波之间不正交
%即就是单个周期内由于时延，导致的一个周期内积分不为零，频谱泄露
signal_time_C = [signal_time(end-CP_length+1:end,:);signal_time];
signal_time_C = [signal_time_C; signal_time_C(1:CS_length,:)]; % 单个完整符号为512+128+20=660

subplot(3,1,2);
plot(signal_time_C(:,1));
xlabel('Time');
ylabel('Amplitude');
title('加CP和CS的单个OFDM符号');
axis([0 500 -0.5 0.5])
% =======================加窗========================
signal_window = zeros(size(signal_time_C));
% 通过矩阵点乘(对应元素相乘)
%100个ofdm符号，每个符号的660个子载波都经过RC窗
temp = repmat(rcoswindow(alpha,size(signal_time_C,1)),1,symbol_count);
signal_window = signal_time_C.*temp;

subplot(3,1,3)
plot(signal_window(:,1))
title('加窗后的单个OFDM符号')
xlabel('Time');
ylabel('Amplitude');
axis([0 500 -0.5 0.5])
% figure(10)
% plot(abs(fft(signal_window(:,1))));

% ===================发送信号，多径信道====================
%时域发送的还是100ofdm符号的连续信号
signal_Tx = reshape(signal_window,1,[]); % 并串转换，变成时域一个完整信号，待传输
signal_origin = reshape(signal_time_C,1,[]); % 未加窗完整信号

%长距离传输幅度会衰减，每个信号到达接收机的时间不同
mult_path_am = [1 0.2 0.1]; %  多径幅度
mutt_path_time = [0 20 50]; % 多径时延
windowed_Tx = zeros(size(signal_Tx));
path2 = 0.2*[zeros(1,20) signal_Tx(1:end-20) ];
path3 = 0.1*[zeros(1,50) signal_Tx(1:end-50) ];
signal_Tx_mult = signal_Tx + path2 + path3; % 多径信号

figure(4)
subplot(2,1,1)
plot(signal_Tx)
title('单径下OFDM信号')
xlabel('Time/samples')
ylabel('Amplitude')
axis([0 1000 -0.5 0.5])

subplot(2,1,2)
plot(signal_Tx_mult)
title('多径下OFDM信号')
xlabel('Time/samples')
ylabel('Amplitude')
axis([0 1000 -0.5 0.5])

% =====================发送信号频谱========================

% 每个符号求频谱再平均，功率取对数
% orgin_aver_power = 20*log10(mean(abs(fft(signal_time_C'))));

% ====================加窗信号频谱=========================

figure(5) % 归一化
%mean函数对每一列求均值,返回每个ofdm信号平均功率
orgin_aver_power = 20*log10(mean(abs(fft(signal_window'))));
plot((1:length(orgin_aver_power))/length(orgin_aver_power),orgin_aver_power)
hold on
axis([0 1 -40 5])
grid on
title('加窗信号频谱')

% ========================加AWGN==========================
%默认信号没有直流分量，均值为零
signal_power_sig = var(signal_Tx); % 单径发送信号功率
signal_power_mut = var(signal_Tx_mult); % 多径发送信号功率

SNR_linear = 10^(SNR/10);
noise_power_mut = signal_power_mut/SNR_linear;
noise_power_sig = signal_power_sig/SNR_linear;
noise_sig = randn(size(signal_Tx))*sqrt(noise_power_sig);
noise_mut = randn(size(signal_Tx_mult))*sqrt(noise_power_mut);

%对发射信号加入了高斯白噪声
Rx_data_sig = signal_Tx+noise_sig;
Rx_data_mut = signal_Tx_mult+noise_mut;

% =======================串并转换==========================

%空白格的意思相当于python中的-1，自己去计算剩余的维度
Rx_data_mut = reshape(Rx_data_mut,ifft_length+CS_length+CP_length,[]);
Rx_data_sig = reshape(Rx_data_sig,ifft_length+CS_length+CP_length,[]);

% ====================去循环前缀和后缀======================
%相当于切片截取部分值
Rx_data_sig(1:CP_length,:) = [];
Rx_data_sig(end-CS_length+1:end,:) = [];
Rx_data_mut(1:CP_length,:) = [];
Rx_data_mut(end-CS_length+1:end,:) = [];

% =========================FFT=============================
%将实数信号恢复成QAM调制后的复数信号
fft_sig = fft(Rx_data_sig);
fft_mut = fft(Rx_data_mut);

% =========================恢复采样===========================
%把对应有数字信息的载波值提取出来
data_sig = fft_sig(carrier_position,:);
data_mut = fft_mut(carrier_position,:);

figure(6)
scatter(real(reshape(data_sig,1,[])),imag(reshape(data_sig,1,[])),'.')
grid on;
title('单径接收信号星座图')

%多径传输导致幅度衰减，相位延迟
figure(7)
scatter(real(reshape(data_mut,1,[])),imag(reshape(data_mut,1,[])),'.')
grid on;
title('多径接收信号星座图')

% =========================16QAM逆映射===========================

bit_demod_sig = reshape(qamdemod(data_sig,16,'OutputType','bit'),[],1);
bit_demod_mut = reshape(qamdemod(data_mut,16,'OutputType','bit'),[],1);

% =========================误码率===========================
error_bit_sig = sum(bit_demod_sig~=bit_sequence);
error_bit_mut = sum(bit_demod_mut~=bit_sequence);
error_rate_sig = error_bit_sig/bit_length;
error_rate_mut = error_bit_mut/bit_length;
rate = [error_rate_sig error_rate_mut]
