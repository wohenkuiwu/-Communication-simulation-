clc
clear

%% 发送机 要求滚降0.25，FS=4M,FS*T=4,1/T=1M,
    M = 16;
    aqam = [-3, -1, 1, 3];
    A = repmat(aqam, 4, 1);
    B = flipud(A');
    konst_qam = A+i*B;
    konst_qam = konst_qam(:); % 产生所有星座点
    qam = konst_qam(randi([0,15], 10000, 1)+1);
    figure()
    plot(qam, 'o');
    title('Ordinary Signal Contellation');
    axis([-4 4 -4 4]);
    xlabel('Re');ylabel('Im')
    
    FS = 3000000;   %采样频率
    N = length(qam);
    T = 1/1000000;
    RS = 1/T;
    FS_T = 3;
    t = -5*T:1/FS:5*T;
    t = t+1e-10;
    alfa = 0.35;
    
    %发送滤波器的时域表达式以及过采样因子缩放
    cositem = cos((1+alfa)*pi*t/T);
    sinitem = sin((1-alfa)*pi*t/T);
    extra = 4*alfa*t/T;
    below = (1-(4*alfa*t/T).^2).*(pi*t/T);
    p = (extra.*cositem+sinitem)./below; %发送滤波器的时域表达
    p = p./(sqrt(FS*T)); %过采样
    
    % 下面是加零和滤波
     qams = zeros(size(1:FS_T*N));
     qams(1:FS_T:FS_T*N) = qam;
     symbols = filter(p, 1, qams);
     
    %下面是滤波器和实际传输信号的幅度谱
    Nfft = 2048;
    P = fftshift(fft(p,Nfft));
    X0 = fftshift(fft(symbols,Nfft));
    f = -FS/2:FS/Nfft:FS/2-FS/Nfft;
    figure()
    subplot(211);
    plot(f, 20*log10(abs(P)));grid;title('Pulse Spectrum in dB');
    xlabel('Freq');
    subplot(212);
    plot(f, 20*log10(abs(X0)));grid;title('Signal Spectrum in dB')
    xlabel('Freq');
    
    figure();
    plot(symbols,'.');  % 信号星座图
    title('Trans Signal Contellation')
    eyediagram(symbols, 2); % 实际发送信号的眼图
%% 通过信道传输
    % 通过线性滤波器
    Fcut = 500000;
    wn_lpf = Fcut*2/FS;
    b_lpf = fir1(4, wn_lpf);
    lpf_symbols = filter(b_lpf, 1, symbols);
    figure()
    X1 = fftshift(fft(lpf_symbols, Nfft));
    plot(f, 20*log10(abs(X1)));grid;title('After Channel Filter Signal Spectrum in dB')
    xlabel('Freq');
    
    % SNR加噪
    noise = randn(size(lpf_symbols))+i*randn(size(lpf_symbols)); % 产生复数噪声
    svsymbol = std(lpf_symbols)^2; 
    nv = std(noise)^2;
    figure()
    Ps = [];
    for SNR = 5:5:20
        p1 = std(lpf_symbols)/(std(noise)*10^(SNR/20));
        sign_withnoise(SNR/5,:) = lpf_symbols+noise*p1;
        Ps(SNR/5,:) = fftshift(fft(sign_withnoise(SNR/5,:),Nfft));
        subplot(2,2,SNR/5)
        plot(f, 20*log10(abs(Ps(SNR/5,:))));grid;title(sprintf("Signal Spectrum in dB in SNR %d",SNR))
        xlabel('Freq');
     end
%% 接收机
    for i = 1:4
        rvfilter(i,:) = filter(p,1,sign_withnoise(i,:));
    end
    % 下面是接受信号的星座图
     for i = 1:4
         eyediagram(rvfilter(i,:),2)
     end
    % 找到相位差,并且重采样
    impulse = [1,zeros(1,100)];
    TxImpOut = filter(p,1,impulse);
    ChannelImpOut = filter(b_lpf, 1, TxImpOut);
    RxImpOut = filter(p,1,ChannelImpOut);
    [Trash, Pos] = max(abs(RxImpOut));
    %RXSAMPOUT能恢复出qam
    figure()
    for i = 1:4
        temp = rvfilter(i,:);
        subplot(4,3,3*i-2)
        plot(rvfilter(i,:), '.')
        title(sprintf("Recieved signal Contellation in SNR %d", i*5))
        subplot(4,3,3*i-1)
        RxSampOut(i,:) = temp(Pos:FS*T:end);
        plot(RxSampOut(i,:),'.')
        title(sprintf("Resampled Signal Contellation in SNR %d", i*5))
    end
%% 信道均衡
    %使用均衡函数进行均衡操作
    sigconst = step(comm.RectangularQAMModulator(M),(0:M-1)');
    eqlms = lineareq(6, lms(0.0008)); % Create an LMS equalizer object.
    eqlms.SigConst = sigconst'; % Set signal constellation.
    eqlms.ResetBeforeFiltering = 0; % Maintain continuity between iterations.
    eq_current = eqlms;
    msglen = length(RxSampOut(1,:));
    modmsg = qam(1:msglen);
    itr = 2000;trainsig = modmsg(1:itr);%取其中1/5训练均衡器抽头系数
    for i = 1:4
        msg = RxSampOut(i,:).';
        y = equalize(eq_current, msg, trainsig);
        subplot(4,3,3*i)
        plot(y(itr+1:end),'.')
        z(i,:) = y;
        title(sprintf("Equalized Signal Contellation in SNR %d", i*5))
    end
    
%% 误码率曲线
for i = 1:4
    sn_block = repmat(z(i,:),16,1);
    konst_block = repmat(konst_qam,1,msglen);
    distance = abs(sn_block-konst_block);
    [dmin,ind_2] = min(distance);
    qam_det = konst_qam(ind_2);
    qamlen = length(qam_det);
    d = 2;
    SNR =5*i;
    p2 = std(qam)/(std(noise)*10^(SNR/20));
    sigma = std(real(noise*p2));
    Q = 0.5*erfc(d/(sqrt(2)*2*sigma));
    sep_theo(i) = 3.5*Q - 3.0625*Q^2;
    number_of_errors = sum(qam(1:qamlen) ~= qam_det);
    sep_simu(i) = number_of_errors/qamlen;
end
    figure()
    A2_over_sigma2_dB=5:5:20;%仿真信噪比范围(dB)
    A2_over_sigma2=10.^(A2_over_sigma2_dB./10);%仿真信号信噪比（倍数）
    hold on; 
    semilogy(A2_over_sigma2_dB,sep_theo,'b');
    semilogy(A2_over_sigma2_dB,sep_simu,'r');
    legend('理论误码率','实际仿真结果')
    title('Bit Error Rate (BER) in SNR')
    xlabel('SNR');ylabel('Bit Error Rate (BER)');
    grid on;
    hold off;

