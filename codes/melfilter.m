function Filter = melfilter(N,FrequencyVector,hWindow)
% melfilter         Create a mel frequency filterbank
%
%   [Filter,MelFrequencyVector] = melfilter(N,FrequencyVector,hWindow)
%
%   Generates a filter bank matrix with N lineary spaced filter banks, 
%   in the Mel frequency domain, that are overlapped by 50%.
%
%   `N` the number of filter banks to construct.
%
%   `FrequencyVector` a vector indicating the frequencies at which to
%   evaluate the filter bank coeffiecents.
%
%   `hWindow` a handle to the windowing function that determines the shape
%   of the filterbank. The default is hWindow = @triang
%
%   `Filter` is sparse matrix of size [N numel(FrequencyVector)].
%
%   `MelFrequencyVector` is a vector containing the Mel frequency values
%
%   Example
%       N = 50;
%       Fs = 10000;
%       x = sin(2*pi*110*(0:(1/Fs):5));
%       [Pxx,F] = periodogram(x,[],512,Fs);
%
%       Filter = melfilter(N,F);
%       Filter = melfilter(N,F,@rectwin);
%       [Filter,MF] = melfilter(N,F,@blackmanharris);
%
%       FPxx = Filter*Pxx;
%
%   See also
%       melfilter melbankm mfcceps hz2mel
%

%% Author Information
%   Pierce Brady
%   Smart Systems Integration Group - SSIG
%	Cork Institute of Technology, Ireland.
% 

%% Reference
%   F. Zheng, G. Zhang, Z. Song, "Comparision of Different Implementations
%   of MFCC", Journal of Computer Science & Technology, vol. 16, no. 6, 
%   September 2001, pp. 582-589
%
%   melbankm by Mike Brookes 1997
%

%% Assign defaults
if nargin<3 || isempty(hWindow), hWindow = @triang; end

%%
MelFrequencyVector = 2595*log10(1+FrequencyVector/700);   % Convert to mel scale
MaxF = max(MelFrequencyVector);                 % 
MinF = min(MelFrequencyVector);                 %
MelBinWidth = (MaxF-MinF)/(N+1);                %
Filter = zeros([N numel(MelFrequencyVector)]);  % Predefine loop matrix

%% Construct filter bank
for i = 1:N
    iFilter = find(MelFrequencyVector>=((i-1)*MelBinWidth+MinF) & ...
                    MelFrequencyVector<=((i+1)*MelBinWidth+MinF));
    Filter(i,iFilter) = hWindow(numel(iFilter)); % Triangle window
end
Filter = sparse(Filter);    % Reduce memory size
figure('Name','Mel frequency filterbank')
for i = 1:N
    plot(Filter(i,:));
    hold on;
end
title('Full 15 Mel Filterbank')
xlabel('Frequency in Hz')
ylabel('Amplitude')
figure('Name','filter 7')
plot(Filter(7,:))
title('Filter 7 from Mel Filterbank')
xlabel('Frequency in Hz')
ylabel('Amplitude')
figure('Name','filter 15')
plot(Filter(15,:))
title('Filter 15 from Mel Filterbank')
xlabel('Frequency in Hz')
ylabel('Amplitude')

end
