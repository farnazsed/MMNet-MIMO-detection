function chan_out = split_snap( h_channel, varargin )
%SPLIT_SNAP Splits channel objects based on snapshot indices.
%
% Calling object:
%   Single object
%
% Description:
%   This function can be used to split a channel object into sub-objects based on a list of
%   snapshots. For example, this can be used to separate channels into LOS and NLOS parts. To split
%   the channels, the following command can be used:     cs = c.split( 1:100, 101:2:200 ); This
%   splits the channel object "c" into two sub-channels, the first containing the snapshots 1 to
%   100, and the second containing the snapshots 101 to 199 (at half resolution). 
% 
% Notes:
%      * Inputs must be scalar channel objects.
%      * If there is evaluation data in the par field, it will be split as well. This requires the
%        field par.cluster_ind which determines the small-scale-fading averaging intervals.
%      * A running index (in the format "p001", "p002", etc.) is added to the channel name, so that
%        the sub-channels can be identified.
%
% Input:
%   varargin
%   A list of snapshot indices. The number of inputs determines the number of output channels.
%
% Output:
%   chan_out
%   The split channel objects
%
%
% QuaDRiGa Copyright (C) 2011-2017 Fraunhofer Heinrich Hertz Institute
% e-mail: quadriga@hhi.fraunhofer.de
%
% QuaDRiGa is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

if numel( h_channel ) > 1
    error('QuaDRiGa:qd_channel:split_snap','??? "split_snap" is only defined for scalar objects.')
else
    h_channel = h_channel(1,1); % workaround for octave
end


splt = varargin;

chan_out = qd_channel;
for i_sp = 1:numel( splt )
    snap = sort( splt{ i_sp } );
    
    % Set coeffs and delays
    if ~isempty( h_channel(1,1).coeff ) && h_channel(1,1).individual_delays
        chan_out( 1,i_sp ) = qd_channel( h_channel(1,1).coeff(:,:,:,snap),...
            h_channel(1,1).delay(:,:,:,snap) );
    elseif ~isempty( h_channel(1,1).coeff ) && ~h_channel(1,1).individual_delays
        chan_out( 1,i_sp ) = qd_channel( h_channel(1,1).coeff(:,:,:,snap),...
            h_channel(1,1).delay(:,snap) );
    else
        chan_out( 1,i_sp ) = qd_channel;
    end
    
    % Set tx position
    chan_out( 1,i_sp ).tx_position = h_channel(1,1).tx_position;
    
    % Set Rx position
    if ~isempty( h_channel(1,1).coeff ) && ~isempty( h_channel(1,1).rx_position )
        chan_out( 1,i_sp ).rx_position = h_channel(1,1).rx_position( :,snap );
    end
    
    % Process par structure
    if ~isempty( h_channel(1,1).par )
        
        if isempty( h_channel(1,1).coeff ) && ~isfield( h_channel(1,1).par, 'cluster_ind' )
            tmp = fieldnames( h_channel(1,1).par );
            tmp = size(h_channel(1,1).par.(tmp{1}),2);
            i_clst = 1:tmp;
        elseif isfield( h_channel(1,1).par, 'cluster_ind' )
            i_clst = h_channel(1,1).par.cluster_ind;
        else
            i_clst = [];
        end
        
        if ~isempty( i_clst )
            
            n_clst = numel( unique( i_clst ) );
            n_snap = numel( i_clst );
            
            % Reduced fields
            fields = fieldnames( h_channel(1,1).par );
            for i_field = 1 : numel( fields )
                dat = h_channel(1,1).par.( fields{ i_field } );
                if size( dat,2 ) == n_snap
                    chan_out( 1,i_sp ).par.( fields{ i_field } ) = dat( :,snap );
                elseif size( dat,2 ) == n_clst
                    clst = unique( i_clst( 1,snap ) );
                    chan_out( 1,i_sp ).par.( fields{ i_field } ) = dat( :,clst );
                end
            end
            
            % Update snapshot indices
            if isfield( chan_out( 1,i_sp ).par, 'cluster_ind' )
                chan_out( 1,i_sp ).par.cluster_snap =...
                    [1 find( diff( chan_out( 1,i_sp ).par.cluster_ind ) )+1];
            end
        end
    end
    
    % Set name
    chan_out( 1,i_sp ).name = [h_channel(1,1).name,'p',num2str(i_sp,'%02d')];
end

end
