function visualize_clusters( h_builder, i_mobile , i_cluster, create_figure )
%VISUALIZE_CLUSTERS Plots the positions of the scattering clusters for a mobile terminal
%
% Calling object:
%   Single object
%
% Description:
%   This method plots all scattering clusters for a given mobile terminal. If i_cluster is not
%   given, then only the main paths are shown for all MPCs. If i_cluster is given, then also the
%   subpaths are shown for the selected cluster. The plot is in 3D coordinates. You can rotate the
%   image using the rotate tool.
%
% Input:
%   i_mobile
%   The index of the mobile terminal within the channel builder object
%
%   i_cluster
%   The index of the scattering cluster. (Optional)
%
%   create_figure
%   If set to 1 (default), a new figure is created. If set to 0, the last figure is updated
%
%
% QuaDRiGa Copyright (C) 2011-2017 Fraunhofer Heinrich Hertz Institute
% e-mail: quadriga@hhi.fraunhofer.de
%
% QuaDRiGa is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.


if numel( h_builder ) > 1
    error('QuaDRiGa:qd_builder:ObjectArray','??? "calc_scatter_positions" is only defined for scalar objects.')
else
    h_builder = h_builder(1,1); % workaround for octave
end

if ~exist( 'i_mobile', 'var' ) || isempty( i_mobile )
    i_mobile = 1;
end

if ~exist( 'i_cluster', 'var' ) || isempty( i_cluster )
    i_cluster = 0;
end

if ~exist( 'create_figure', 'var' ) || isempty( create_figure )
    create_figure = true;
end


NumSubPaths = h_builder.NumSubPaths;

tx_pos = h_builder.tx_position;
rx_pos = h_builder.rx_positions(:,i_mobile);

[ ~, ~ , lbs_pos , fbs_pos ] = h_builder.calc_scatter_positions( i_mobile );
nNLOS = h_builder.NumClusters;

lbs_center = qf.clst_avg( lbs_pos, NumSubPaths );
fbs_center = qf.clst_avg( fbs_pos, NumSubPaths );

[ ahat_x, ahat_y, ahat_z ] = sph2cart(h_builder.AoA(i_mobile,:), h_builder.EoA(i_mobile,:), 1);
ahat_x = ahat_x + rx_pos(1);
ahat_y = ahat_y + rx_pos(2);
ahat_z = ahat_z + rx_pos(3);

[ bhat_x, bhat_y, bhat_z ] = sph2cart(h_builder.AoD(i_mobile,:), h_builder.EoD(i_mobile,:), 1);
bhat_x = bhat_x + tx_pos(1);
bhat_y = bhat_y + tx_pos(2);
bhat_z = bhat_z + tx_pos(3);

if any( abs( lbs_center(:) - fbs_center(:)) > 1e-11 )
    show_FBS_LBS_path = true;
else
    show_FBS_LBS_path = false;
end

if create_figure
    figure('Position',[ 100 , 100 , 1000 , 700]);
end

% Plot Tx position
plot3( tx_pos(1),tx_pos(2),tx_pos(3),'sr','Linewidth',2,'Markersize',10);
hold on
text( tx_pos(1),tx_pos(2),tx_pos(3),'Tx' );

% Plot Rx position
plot3( rx_pos(1),rx_pos(2),rx_pos(3),'ob','Linewidth',2,'Markersize',10 );
text( rx_pos(1),rx_pos(2),rx_pos(3),'Rx' );

% Plot First LBS and FBS position
if i_cluster ~= 0
    for i_sub = 1:numel( i_cluster )
        i_cl = i_cluster( i_sub );
        plot3( fbs_center(1,i_cl),fbs_center(2,i_cl),fbs_center(3,i_cl) ,...
            '^r','Linewidth',1,'Markersize',8,'Markerfacecolor','r' );
        plot3( lbs_center(1,i_cl),lbs_center(2,i_cl),lbs_center(3,i_cl) ,...
            'vb','Linewidth',1,'Markersize',8,'Markerfacecolor','b' );
        
        % Plot path from Tx to FBS
        plot3( [tx_pos(1),fbs_center(1,i_cl)],...
            [tx_pos(2),fbs_center(2,i_cl)],...
            [tx_pos(3),fbs_center(3,i_cl)] ,'-.r','Linewidth',2 );
        
        % Plot path from Rx to LBS
        plot3( [rx_pos(1),lbs_center(1,i_cl)],...
            [rx_pos(2),lbs_center(2,i_cl)],...
            [rx_pos(3),lbs_center(3,i_cl)] ,'-.b','Linewidth',2 );
        
        % Plot path from FBS to LBS
        if show_FBS_LBS_path
            plot3( [fbs_center(1,i_cl),lbs_center(1,i_cl)],...
                [fbs_center(2,i_cl),lbs_center(2,i_cl)],...
                [fbs_center(3,i_cl),lbs_center(3,i_cl)] ,'-.m','Linewidth',2 );
        end
    end
    
else
    i_cl = 2;
    plot3( fbs_center(1,i_cl),fbs_center(2,i_cl),fbs_center(3,i_cl) ,...
        '+r','Linewidth',2,'Markersize',7 );
    plot3( lbs_center(1,i_cl),lbs_center(2,i_cl),lbs_center(3,i_cl) ,...
        'ob','Linewidth',2,'Markersize',7 );
    
    % Plot path from Tx to FBS
    plot3( [tx_pos(1),fbs_center(1,i_cl)],...
        [tx_pos(2),fbs_center(2,i_cl)],...
        [tx_pos(3),fbs_center(3,i_cl)] ,'-.r','Linewidth',1 );
    
    % Plot path from Rx to LBS
    plot3( [rx_pos(1),lbs_center(1,i_cl)],...
        [rx_pos(2),lbs_center(2,i_cl)],...
        [rx_pos(3),lbs_center(3,i_cl)] ,'-.b','Linewidth',1 );
    
    % Plot path from FBS to LBS
    if show_FBS_LBS_path
        plot3( [fbs_center(1,i_cl),lbs_center(1,i_cl)],...
            [fbs_center(2,i_cl),lbs_center(2,i_cl)],...
            [fbs_center(3,i_cl),lbs_center(3,i_cl)] ,'-.m','Linewidth',1 );
    end
end

if i_cluster ~= 0
    for i_sub = 1:numel( i_cluster )
        i_cl = i_cluster( i_sub );
        if all( abs( lbs_center(:,i_cl) - fbs_center(:,i_cl)) < 1e-11 )
            text( lbs_center(1,i_cl),lbs_center(2,i_cl),lbs_center(3,i_cl) ,...
                ['FL-',num2str(i_cl,'%02d')]);
        else
            text( lbs_center(1,i_cl),lbs_center(2,i_cl),lbs_center(3,i_cl) ,...
                ['L-',num2str(i_cl,'%02d')]);
            text( fbs_center(1,i_cl),fbs_center(2,i_cl),fbs_center(3,i_cl) ,...
                ['F-',num2str(i_cl,'%02d')]);
        end
    end
end

% Plot r0, the vector pointing from Tx to Rx
plot3( [tx_pos(1),rx_pos(1)],[tx_pos(2),rx_pos(2)],[tx_pos(3),rx_pos(3)],...
    '-k','Linewidth',2);

% Plot first normalized angle
plot3( [rx_pos(1) ahat_x(i_cl)],[rx_pos(2) ahat_y(i_cl)],[rx_pos(3) ahat_z(i_cl)],...
    '-ob','Linewidth',2,'Markersize',10,'Markerfacecolor','y' );
plot3( [tx_pos(1) bhat_x(i_cl)],[tx_pos(2) bhat_y(i_cl)],[tx_pos(3) bhat_z(i_cl)],...
    '-sr','Linewidth',2,'Markersize',10,'Markerfacecolor','y' );

if i_cluster == 0
    for i_cl = 3:nNLOS
        
        % Plot First LBS and FBS position
        plot3( fbs_center(1,i_cl),fbs_center(2,i_cl),fbs_center(3,i_cl) ,...
            '+r','Linewidth',2,'Markersize',7 );
        plot3( lbs_center(1,i_cl),lbs_center(2,i_cl),lbs_center(3,i_cl) ,...
            'ob','Linewidth',2,'Markersize',7 );
        
        % Plot path from Tx to FBS
        plot3( [tx_pos(1),fbs_center(1,i_cl)],...
            [tx_pos(2),fbs_center(2,i_cl)],...
            [tx_pos(3),fbs_center(3,i_cl)] ,'-.r','Linewidth',1 );
        
        % Plot path from Rx to LBS
        plot3( [rx_pos(1),lbs_center(1,i_cl)],...
            [rx_pos(2),lbs_center(2,i_cl)],...
            [rx_pos(3),lbs_center(3,i_cl)] ,'-.b','Linewidth',1 );
        
        % Plot path from FBS to LBS
        if show_FBS_LBS_path
            plot3( [fbs_center(1,i_cl),lbs_center(1,i_cl)],...
                [fbs_center(2,i_cl),lbs_center(2,i_cl)],...
                [fbs_center(3,i_cl),lbs_center(3,i_cl)] ,'-.m','Linewidth',1 );
        end
    end
    
    % Plot the arrival angles
    for i_cl = [ 1,3:nNLOS ]
        plot3( [rx_pos(1) ahat_x(i_cl)],[rx_pos(2) ahat_y(i_cl)],[rx_pos(3) ahat_z(i_cl)],...
            '-ob','Linewidth',2,'Markersize',10,'Markerfacecolor','y' );
        plot3( [tx_pos(1) bhat_x(i_cl)],[tx_pos(2) bhat_y(i_cl)],[tx_pos(3) bhat_z(i_cl)],...
            '-sr','Linewidth',2,'Markersize',10,'Markerfacecolor','y' );
    end
    
else
    for i_path = 1:numel( i_cluster )
        i_cl = i_cluster( i_path );
        fbs_pos_cl = qf.clst_extract( fbs_pos, NumSubPaths, i_cl );
        lbs_pos_cl = qf.clst_extract( lbs_pos, NumSubPaths, i_cl );

        for i_sub = 1:NumSubPaths( i_cl );
            
            % Plot First LBS and FBS position
            plot3( fbs_pos_cl(1,i_sub),fbs_pos_cl(2,i_sub),fbs_pos_cl(3,i_sub) ,...
                '+r','Linewidth',2,'Markersize',7 );
            plot3( lbs_pos_cl(1,i_sub),lbs_pos_cl(2,i_sub),lbs_pos_cl(3,i_sub) ,...
                'ob','Linewidth',2,'Markersize',7 );
            
            % Plot path from Tx to FBS
            plot3( [tx_pos(1),fbs_pos_cl(1,i_sub)],...
                [tx_pos(2),fbs_pos_cl(2,i_sub)],...
                [tx_pos(3),fbs_pos_cl(3,i_sub)] ,':r' );
            
            % Plot path from Tx to FBS
            plot3( [rx_pos(1),lbs_pos_cl(1,i_sub)],...
                [rx_pos(2),lbs_pos_cl(2,i_sub)],...
                [rx_pos(3),lbs_pos_cl(3,i_sub)] ,':b' );
            
            % Plot path from Tx to FBS
            plot3( [fbs_pos_cl(1,i_sub),lbs_pos_cl(1,i_sub)],...
                [fbs_pos_cl(2,i_sub),lbs_pos_cl(2,i_sub)],...
                [fbs_pos_cl(3,i_sub),lbs_pos_cl(3,i_sub)] ,':m' );
        end
    end
end

% Non-Legend Elements
% Line from ground to Tx pos
plot3( [tx_pos(1),tx_pos(1)] , ...
    [tx_pos(2),tx_pos(2)],...
    [0,tx_pos(3)] ,'-r','Linewidth',3 );

plot3( [rx_pos(1),rx_pos(1)] , ...
    [rx_pos(2),rx_pos(2)],...
    [0,rx_pos(3)] ,'-b','Linewidth',3 );

hold off
grid on
box on

view(0, 90);

xlabel('x-coords in [m]');
ylabel('y-coords in [m]');
zlabel('z-coords in [m]');

if show_FBS_LBS_path
    legend('Tx-Position','Rx-Position','FBS','LBS','Tx-FBS (b)',...
        'Rx-LBS (a)','FBS-LBS (c)','LOS Path',...
        'Rx-Angle','Tx-Angle',...
        'Location','NorthEastOutside')
else
    legend('Tx-Position','Rx-Position','FBS','LBS','Tx-FBS (b)',...
        'Rx-LBS (a)','LOS Path',...
        'Rx-Angle','Tx-Angle',...
        'Location','NorthEastOutside')
end

a = axis;
dx = a(2)-a(1);
mx = a(1)+0.5*dx;
dy = a(4)-a(3);
my = a(3)+0.5*dy;

if dx > dy
    axis([a(1) a(2) my-0.5*dx my+0.5*dx]);
else
    axis([mx-0.5*dy mx+0.5*dy a(3) a(4)]);
end

end
