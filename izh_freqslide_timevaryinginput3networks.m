% code for figure 2e from:
% Cohen, M.X (2014). Fluctuations in oscillation frequency control spike
% timing and coordinate neural networks. Journal of Neuroscience

% This code was based on code from Izhikevich, February 25, 2003


clear

%% initialize...

% number of neurons
Nexcit = 800;
Ninhib = 200;

srate    = 1000; % Hz
stim_dur = 5000; % ms
nTrials  =  20;

inputstrength = 15; % arb. units


corfreqslide = zeros(nTrials,3);

re1 = rand(Nexcit,1); ri1 = rand(Ninhib,1);
re2 = rand(Nexcit,1); ri2 = rand(Ninhib,1);
re3 = rand(Nexcit,1); ri3 = rand(Ninhib,1);

gam1 = zeros(nTrials,stim_dur-1);
gam2 = zeros(nTrials,stim_dur-1);
gam3 = zeros(nTrials,stim_dur-1);

%% median filter parameters

n_order = 10;
orders = round(linspace(10,400,n_order)); % recommended: 10 steps between 10 and 400 ms (hard coded b/c srate=1000)
orders = floor((orders-1)/2); % pre/post halves
phasedmed1 = zeros(length(orders),stim_dur-1);
phasedmed2 = zeros(length(orders),stim_dur-1);
phasedmed3 = zeros(length(orders),stim_dur-1);

%%

for trials = 1:nTrials
    
    % voltage
    voltages1 = -65*ones(Nexcit+Ninhib,1); % Initial values of v
    voltages2 = -65*ones(Nexcit+Ninhib,1); % Initial values of v
    voltages3 = -65*ones(Nexcit+Ninhib,1); % Initial values of v
    
    a1 = [ 0.02*ones(Nexcit,1); 0.02+0.08*ri1 ];
    a2 = [ 0.02*ones(Nexcit,1); 0.02+0.08*ri2 ];
    a3 = [ 0.02*ones(Nexcit,1); 0.02+0.08*ri3 ];
    b1 = [ 0.20*ones(Nexcit,1); 0.25-0.05*ri1 ];
    b2 = [ 0.20*ones(Nexcit,1); 0.25-0.05*ri2 ];
    b3 = [ 0.20*ones(Nexcit,1); 0.25-0.05*ri3 ];
    restingVoltage1 = [-65+15*re1.^2; -65*ones(Ninhib,1)]; % aka: c
    restingVoltage2 = [-65+15*re2.^2; -65*ones(Ninhib,1)]; % aka: c
    restingVoltage3 = [-65+15*re3.^2; -65*ones(Ninhib,1)]; % aka: c
    d1 = [ 8-6*re1.^2; 2*ones(Ninhib,1)];
    d2 = [ 8-6*re2.^2; 2*ones(Ninhib,1)];
    d3 = [ 8-6*re3.^2; 2*ones(Ninhib,1)];
    S1 = [0.5*rand(Nexcit+Ninhib,Nexcit), -rand(Nexcit+Ninhib,Ninhib)];
    S2 = [0.5*rand(Nexcit+Ninhib,Nexcit), -rand(Nexcit+Ninhib,Ninhib)];
    S3 = [0.5*rand(Nexcit+Ninhib,Nexcit), -rand(Nexcit+Ninhib,Ninhib)];
    u1 = b1.*voltages1; % Initial values of u
    u2 = b2.*voltages2; % Initial values of u
    u3 = b3.*voltages3; % Initial values of u
    
    
    % initialize firing matrix
    fireall1 = zeros(Nexcit+Ninhib,stim_dur*(srate/1000));
    fireall2 = zeros(Nexcit+Ninhib,stim_dur*(srate/1000));
    fireall3 = zeros(Nexcit+Ninhib,stim_dur*(srate/1000));
    firings1 = []; % spike timings
    firings2 = []; % spike timings
    firings3 = []; % spike timings
    
    % initial firings
    fired1 = voltages1 >= 30; % indices of spikes
    fired2 = voltages2 >= 30; % indices of spikes
    fired3 = voltages3 >= 30; % indices of spikes
    
    
    % define long-range input
    inputstrength2use = inputstrength+.2*(inputstrength+5)*sin(2*pi*.5*(1:stim_dur)/srate);
    inputzAll1 = [ bsxfun(@times,inputstrength2use,randn(Nexcit,stim_dur)); bsxfun(@times,inputstrength2use/2.5,randn(Ninhib,stim_dur)) ];
    inputzAll2 = [ bsxfun(@times,inputstrength2use,randn(Nexcit,stim_dur)); bsxfun(@times,inputstrength2use/2.5,randn(Ninhib,stim_dur)) ];
    inputzAll3 = [ bsxfun(@times,inputstrength2use,randn(Nexcit,stim_dur)); bsxfun(@times,inputstrength2use/2.5,randn(Ninhib,stim_dur)) ];
    
    %%
    
    for timei=1:stim_dur*(srate/1000)
        
        % add local input
        input1 = inputzAll1(:,timei) + 1.2*sum(S1(:,fired1),2);
        input2 = inputzAll1(:,timei) +  .6*sum(S1(:,fired1),2) + .6*sum(S2(:,fired2),2);
        input3 = inputzAll1(:,timei) + 1.2*sum(S3(:,fired3),2);
        
        
        % update voltages (at twice sampling rate for numerical stability)
        voltages1 = voltages1 + .5*(.04*voltages1.^2 + 5*voltages1 + 140 - u1 + input1);
        voltages1 = voltages1 + .5*(.04*voltages1.^2 + 5*voltages1 + 140 - u1 + input1);
        voltages2 = voltages2 + .5*(.04*voltages2.^2 + 5*voltages2 + 140 - u2 + input2);
        voltages2 = voltages2 + .5*(.04*voltages2.^2 + 5*voltages2 + 140 - u2 + input2);
        voltages3 = voltages3 + .5*(.04*voltages3.^2 + 5*voltages3 + 140 - u3 + input3);
        voltages3 = voltages3 + .5*(.04*voltages3.^2 + 5*voltages3 + 140 - u3 + input3);
        
        u1 = u1 + a1.*(b1.*voltages1-u1);
        u2 = u2 + a2.*(b2.*voltages2-u2);
        u3 = u3 + a3.*(b3.*voltages3-u3);
        
        % find which neurons spike
        fired1   = find(voltages1>=30); % indices of spikes
        fired2   = find(voltages2>=30); % indices of spikes
        fired3   = find(voltages3>=30); % indices of spikes
        %firings1 = [firings1; timei+0*fired1,fired1];
        %firings2 = [firings2; timei+0*fired2,fired2];
        %firings3 = [firings3; timei+0*fired3,fired3];
        
        
        fireall1(:,timei) = voltages1;
        fireall2(:,timei) = voltages2;
        fireall3(:,timei) = voltages3;
        
        % reset voltages of fired neurons
        voltages1(fired1)  = restingVoltage1(fired1);
        voltages2(fired2)  = restingVoltage2(fired2);
        voltages3(fired3)  = restingVoltage3(fired3);
        u1(fired1)         = u1(fired1) + d1(fired1);
        u2(fired2)         = u2(fired2) + d2(fired2);
        u3(fired3)         = u3(fired3) + d3(fired3);
    end
    
    %% frequency sliding (temporal derivative of phase angle time series)
    
    gamhil1 = hilbert(eegfilt(mean(fireall1(1:Nexcit,:),1),srate,40,90));
    gamhil2 = hilbert(eegfilt(mean(fireall2(1:Nexcit,:),1),srate,40,90));
    gamhil3 = hilbert(eegfilt(mean(fireall3(1:Nexcit,:),1),srate,40,90));
    
    phased1 = diff(unwrap(angle(gamhil1)));
    phased2 = diff(unwrap(angle(gamhil2)));
    phased3 = diff(unwrap(angle(gamhil3)));
    
    %% median filter
    
    for oi=1:n_order
        for ti=1:length(phased1)
            temp = sort(phased1( max(ti-orders(oi),1):min(ti+orders(oi),stim_dur-1) ));
            phasedmed1(oi,ti) = temp(floor(numel(temp)/2)+1);
            
            temp = sort(phased2( max(ti-orders(oi),1):min(ti+orders(oi),stim_dur-1) ));
            phasedmed2(oi,ti) = temp(floor(numel(temp)/2)+1);
            
            temp = sort(phased3( max(ti-orders(oi),1):min(ti+orders(oi),stim_dur-1) ));
            phasedmed3(oi,ti) = temp(floor(numel(temp)/2)+1);
        end
    end
    
    gam1(trials,:) = srate*mean(phasedmed1,1)'/(2*pi);
    gam2(trials,:) = srate*mean(phasedmed2,1)'/(2*pi);
    gam3(trials,:) = srate*mean(phasedmed3,1)'/(2*pi);
    
    %% compute correlated frequency sliding
    
    corfreqslide(trials,1) = corr(gam1(trials,:)',gam2(trials,:)');
    corfreqslide(trials,2) = corr(gam1(trials,:)',gam3(trials,:)');
    corfreqslide(trials,3) = corr(gam2(trials,:)',gam3(trials,:)');
    
end % end trials

%% plotting...

figure, clf
plot(1,corfreqslide(:,1),'o')
hold on
plot(2,corfreqslide(:,2),'o')
plot(3,corfreqslide(:,3),'o')
set(gca,'xlim',[0 4],'xtick',1:3,'xticklabel',{'1-2';'1-3';'2-3'},'ylim',[.2 1])
xlabel('Network pair'), ylabel('Correlation coefficient')

%%
