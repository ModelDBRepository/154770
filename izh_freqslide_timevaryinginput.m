% code for figure 2c from:
% Cohen, M.X (2014). Fluctuations in oscillation frequency control spike
% timing and coordinate neural networks. Journal of Neuroscience

% This code was based on code from Izhikevich, February 25, 2003

clear

% number of neurons
Nexcit = 800;
Ninhib = 200;

srate    = 1000; % Hz
stim_dur = 5000; % ms
inputstrength = 10;

re=rand(Nexcit,1);
ri=rand(Ninhib,1);

nTrials = 20;

gam = zeros(nTrials,stim_dur-1);

%% median filter parameters

n_order = 10;
orders = round(linspace(10,400,n_order)); % recommended: 10 steps between 10 and 400 ms
orders = floor((orders-1)/2); % pre/post halves
phasedmed = zeros(length(orders),stim_dur-1);

%%

for trials=1:nTrials
    
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
    
    % define long-range input
    inputstrength2use = inputstrength+.2*(inputstrength+5)*sin(2*pi*.5*(1:stim_dur)/srate);
    inputzAll = [ bsxfun(@times,inputstrength2use,randn(Nexcit,stim_dur)); bsxfun(@times,inputstrength2use/2.5,randn(Ninhib,stim_dur)) ];
    
    for timei=1:stim_dur*(srate/1000)
        
        % add local input
        inputz = inputzAll(:,timei) + 1.2*sum(S(:,fired),2);
        
        
        % update voltages (at twice sampling rate for numerical stability)
        voltages = voltages + .5*(.04*voltages.^2 + 5*voltages + 140 - u + inputz);
        voltages = voltages + .5*(.04*voltages.^2 + 5*voltages + 140 - u + inputz);
        
        u = u + a.*(b.*voltages-u);
        
        % find which neurons spike
        fired = find(voltages>=30); % indices of spikes
        %firings=[firings; timei+0*fired,fired];
        
        
        % reset voltages of fired neurons
        voltages(fired)  = restingVoltage(fired);
        u(fired)         = u(fired) + d(fired);
        
        fireall(:,timei) = voltages;
    end
    
    phased = diff(unwrap(angle(hilbert(eegfilt(mean(fireall(1:Nexcit,:),1),srate,40,90)))));
    
    %% median filter
    
    for oi=1:n_order
        for ti=1:length(phased)
            temp = sort(phased( max(ti-orders(oi),1):min(ti+orders(oi),stim_dur-1) ));
            phasedmed(oi,ti) = temp(floor(numel(temp)/2)+1);
        end
    end
    
    gam(trials,:) = srate*mean(phasedmed,1)'/(2*pi);
    
    %% end median filter
    
end % end trials

%%

figure
plot(gam','k')

xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title('Peak frequency as function of oscillating input')

%%


