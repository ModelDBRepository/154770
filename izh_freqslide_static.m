% code for figure 2a-b from:
% Cohen, M.X (2014). Fluctuations in oscillation frequency control spike
% timing and coordinate neural networks. Journal of Neuroscience

% This code was based on code from Izhikevich, February 25, 2003

clear

%% setup

totalN       = 1000; % total number of neurons
pInhibitory  =  0.2; % percent of inhibitory neurons
Iinstrengths = 1:10; % input strengths (arbitrary units)
ntrials      =  100; 

srate    = 1000; % Hz
stim_dur = 5000; % ms


lfp = zeros(length(Iinstrengths),stim_dur);
ftz = zeros(length(Iinstrengths),stim_dur);

%% run simulation

for trials=1:ntrials
    
    % define proportion of E/I
    Nexcit = round(totalN*(1-pInhibitory));
    Ninhib = totalN-Nexcit;
    
    re = rand(Nexcit,1);
    ri = rand(Ninhib,1);
    
    
    for inputstrength=1:length(Iinstrengths)
        
        % voltage
        voltages = -65*ones(Nexcit+Ninhib,1); % Initial values of v
        
        a = [ 0.02*ones(Nexcit,1); 0.02+0.08*ri ];
        b = [ 0.20*ones(Nexcit,1); 0.25-0.05*ri ];
        restingVoltage = [-65+15*re.^2; -65*ones(Ninhib,1)]; % aka: c
        d = [ 8-6*re.^2; 2*ones(Ninhib,1)];
        S = [0.5*rand(Nexcit+Ninhib,Nexcit), -rand(Nexcit+Ninhib,Ninhib)];
        u = b.*voltages; % Initial values of u
        
        
        % initialize firing matrix
        fireall = zeros(Nexcit+Ninhib,stim_dur*(srate/1000));
        firings = []; % spike timings
        
        % initial firings
        fired = voltages >= 30; % indices of spikes
        
        
        % define long-range input (take out of time loop!)
        inputz = [ (inputstrength+5)*randn(Nexcit,stim_dur*(srate/1000)+199); ((inputstrength+5)/2.5)*randn(Ninhib,stim_dur*(srate/1000)+199) ]; % orig 5
        
        for timei=1:stim_dur*(srate/1000)+199 % throw out first 200 ms
            
            % add local input
            Iin = inputz(:,timei) + 1.2*sum(S(:,fired),2);
            
            % update voltages (at twice sampling rate for numerical stability)
            voltages = voltages + .5*(.04*voltages.^2 + 5*voltages + 140 - u + Iin);
            voltages = voltages + .5*(.04*voltages.^2 + 5*voltages + 140 - u + Iin);
            
            u = u + a.*(b.*voltages-u);
            
            % find which neurons spike
            fired = find(voltages>=30); % indices of spikes
            %firings=[firings; timei+0*fired,fired];
            
            % reset voltages of fired neurons
            voltages(fired)  = restingVoltage(fired);
            u(fired)         = u(fired) + d(fired);
            
            fireall(:,timei) = voltages;
        end
        
        %lfp(pInh,inputstrength,:) = mean(fireall(1:Nexcit,:),1);
        ftz(inputstrength,:) = squeeze(ftz(inputstrength,:)) + 2*abs(fft(detrend( mean(fireall(1:Nexcit,200:end),1)'))'/length(ftz));
        
    end % end input strengths
end % end trials

% plot(firings(:,1),firings(:,2),'.');
% plot(mean(fireall(1:Nexcit,:),1))

ftz = ftz/trials;
f = srate/2*linspace(0,1,size(ftz,2)/2+1);

%% plotting...

% gamma power
powGamma  = dsearchn(f',[35 90]');
avegampow = mean(ftz(:,powGamma(1):powGamma(2)),2);

% alpha power
powAlpha  = dsearchn(f',[5 15]');
avealppow = mean(ftz(:,powAlpha(1):powAlpha(2)),2);

% power at peak frequency
freqpeak = zeros(2,size(ftz,1),2);
ff       = zeros(size(ftz,1),length(ftz)/2+1);

% Gaussian to smooth FFT
gauss = exp(-(-10:20/99:10).^2);


for iIni=1:size(ftz,1)
    [~,tmppeak] = max(ftz(iIni,powGamma(1):powGamma(2)));
    tmppeak = tmppeak+powGamma(1)-1;
    freqpeak(1,iIni,1) = f(tmppeak);
    freqpeak(1,iIni,2) = mean(ftz(iIni,tmppeak-10:tmppeak+10),2);
    
    
    [~,tmppeak] = max(ftz(iIni,powAlpha(1):powAlpha(2)));
    tmppeak = tmppeak+powAlpha(1)-1;
    freqpeak(2,iIni,1) = f(tmppeak);
    freqpeak(2,iIni,2) = mean(ftz(iIni,tmppeak-10:tmppeak+10),2);
    
    ff(iIni,:) = conv(squeeze(ftz(iIni,1:length(ftz)/2+1)),gauss,'same');
end


figure(1), clf
subplot(221)
plot((1:10)+5,avegampow,'r-o')
hold on
plot((1:10)+5,freqpeak(1,:,2),'-p')
xlabel('Input strength (a.u.)'), ylabel('Power')
title('Gamma power')
legend({'average';'at peak'})
set(gca,'ylim',[0.05 .3],'xlim',[5 16],'xtick',6:16)

subplot(222)
plot((1:10)+5,freqpeak(1,:,1),'-o')
title('Gamma peak frequency')
xlabel('Input strength (a.u.)'), ylabel('Frequency (Hz)')
set(gca,'ylim',[37 82],'xlim',[5 16],'xtick',6:16)


subplot(223)
plot((1:10)+5,avealppow,'r-o')
hold on
plot((1:10)+5,freqpeak(2,:,2),'-p')
xlabel('Input strength (a.u.)'), ylabel('Power')
title('Alpha power')
legend({'average';'at peak'})
set(gca,'ylim',[0.05 .3],'xlim',[5 16],'xtick',6:16)

subplot(224)
plot((1:10)+5,freqpeak(2,:,1),'-o')
title('Alpha peak frequency')
xlabel('Input strength (a.u.)'), ylabel('Frequency (Hz)')
set(gca,'ylim',[9 16],'xlim',[5 16],'xtick',6:16)

%% power spectra for different input strengths

figure(2), clf
plot(f,ff)
set(gca,'xlim',[0 120])
xlabel('Frequency (Hz)'), ylabel('Power')

%%

