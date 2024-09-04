classdef qd_arrayant < handle
%QD_ARRAYANT QuaDRiGa array antenna class
%
% DESCRIPTION
% This class combines all functions to create and edit array antennas. An array
% antenna is a set of single antenna elements, each having a specific beam pattern,
% that can be combined in any geometric arrangement. A set of synthetic arrays
% that allow simulations without providing your own antenna patterns is
% provided (see generate method for more details).
%
% REFERENCE
% The main functionality was taken from the Winner channel model. "Kyösti, P.;
% Meinilä, J.; Hentilä, L. & others; {IST-4-027756 WINNER II D1.1.2 v.1.1}:
% WINNER II Channel Models; 2007". New functionality has been added to provide
% geometric polarization calculations and antenna pattern rotations.
%
% EXAMPLE
% This example creates an array antenna of crossed dipoles.
%
%    a = qd_arrayant('dipole');           % Create new qd_arrayant object
%    a.copy_element(1,2);                 % Duplicate the dipole
%    a.rotate_pattern(90,'y',2);          % Rotate the second element by 90°
%    a.visualize;                         % Show the output
%
%
% QuaDRiGa Copyright (C) 2011-2017 Fraunhofer Heinrich Hertz Institute
% e-mail: quadriga@hhi.fraunhofer.de
% 
% Fraunhofer Heinrich Hertz Institute
% Wireless Communication and Networks
% Einsteinufer 37, 10587 Berlin, Germany
%  
% This file is part of QuaDRiGa.
% 
% QuaDRiGa is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published 
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% QuaDRiGa is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
%     
% You should have received a copy of the GNU Lesser General Public License
% along with QuaDRiGa. If not, see <http://www.gnu.org/licenses/>.
    
    properties
        name = 'New array';                     % Name of the array antenna
    end
    
    properties(Dependent)
        % Number of antenna elements in the array antenna
        %    Increasing the number of elements creates new elements which
        %    are initialized as copies of the first element. Decreasing the
        %    number of elements deletes the last elements from the array antenna.
        no_elements
        
        % Elevation angles in [rad] were samples of the field patterns are provided
        %   The field patterns are given in spherical coordinates. This
        %   variable provides the elevation sampling angles in radians
        %   ranging from -pi/2 (downwards) to pi/2 (upwards).
        elevation_grid

        % Azimuth angles in [rad] were samples of the field patterns are provided
        %   The field patterns are given in spherical coordinates. This
        %   variable provides the azimuth sampling angles in radians
        %   ranging from -pi to pi.
        azimuth_grid
        
        % Position of the antenna elements in local Cartesian coordinates (units of [m])
        element_position
        
        % The first component of the antenna pattern
        %   contains the vertical component of the electric field given in spherical
        %   coordinates (aligned with the phi direction of the coordinate system). This
        %   variable is a tensor with dimensions [ elevation, azimuth, element-index ]
        %   describing the e-phi component of the far field of each antenna element in the
        %   array.
        Fa
        
        % The second component of the antenna pattern 
        %   contains the horizontal component of the electric field given in spherical
        %   coordinates (aligned with the theta direction of the coordinate system). This
        %   variable is a tensor with dimensions [ elevation, azimuth, element-index ]
        %   describing the e-theta component of the far field of each antenna element in
        %   the array.
        Fb
        
        % Coupling matrix between elements
        %   This matrix describes a pre- or post-processing of the signals
        %   that are fed to the antenna elements. For example, in order to
        %   transmit a LHCP signal, two antenna elements are needed.
        %   They are then coupled by a matrix
        %
        %   	1/sqrt(2) * [1;j]
        %
        %   The rows in the matrix correspond to the antenna elements, the
        %   columns to the signal ports. In this example, the antenna has
        %   one port, i.e. it is fed with one input signal. This signal is
        %   then split into two and fed to the two antenna elements where
        %   the second element radiates the signal with 90 degree phase
        %   shift.
        %   In a similar fashion, it is possible to create fixed
        %   beamforming antennas and include crosstalk between antenna
        %   elements. By default, coupling is set to an identity matrix
        %   which indicates perfect isolation between the antenna elements.           
        coupling
        
        no_az                              % Number of azimuth values
        no_el                              % Number of elevation values
    end
    
    properties(Access=private)
        Pno_elements                = 1;
        Pelevation_grid             = [ -1.570796326794897,0,1.570796326794897];
        Pazimuth_grid               = [ -3.141592653589793,-1.570796326794897,0,...
            1.570796326794897,3.141592653589793];
        Pelement_position           = [0;0;0];
        PFa                         = ones(3,5);
        PFb                         = zeros(3,5);
        Pcoupling                   = 1;
    end
    
    properties(Hidden)
        OctEq = false; % For qf.eq_octave
    end
    
    methods

        function h_qd_arrayant = qd_arrayant( array_type, varargin )
        %QD_ARRAYANT Creates a new array object. 
        %
        % Calling object:
        %   None (constructor)
        %
        % Description:
        %   The constructor calls qd_arrayant.generate to create new array antennas.
        %   If no input is specified, an vertically polarized omni-antenna is
        %   generated. See qd_arrayant.generate for a description of the input
        %   parameters and the list of supported antenna types.
        %
        %
        % QuaDRiGa Copyright (C) 2011-2017 Fraunhofer Heinrich Hertz Institute
        % e-mail: quadriga@hhi.fraunhofer.de
        %
        % QuaDRiGa is free software: you can redistribute it and/or modify
        % it under the terms of the GNU Lesser General Public License as published
        % by the Free Software Foundation, either version 3 of the License, or
        % (at your option) any later version.
            if nargin > 0
                if ~isempty( array_type )
                    h_qd_arrayant = qd_arrayant.generate( array_type , varargin{:} );
                end
            else
                h_qd_arrayant = qd_arrayant.generate( 'omni' );
            end
        end
        
        % Get functions
        function out = get.no_elements(h_qd_arrayant)
            out = h_qd_arrayant.Pno_elements;
        end
        function out = get.elevation_grid(h_qd_arrayant)
            out = h_qd_arrayant.Pelevation_grid; % Single
        end
        function out = get.azimuth_grid(h_qd_arrayant)
            out = h_qd_arrayant.Pazimuth_grid; % Single
        end
        function out = get.element_position(h_qd_arrayant)
            out = h_qd_arrayant.Pelement_position; % Single
        end
        function out = get.Fa(h_qd_arrayant)
            out = h_qd_arrayant.PFa; % Single
        end
        function out = get.Fb(h_qd_arrayant)
            out = h_qd_arrayant.PFb; % Single
        end
        function out = get.coupling(h_qd_arrayant)
            out = h_qd_arrayant.Pcoupling; % Single
        end
        function out = get.no_az(h_qd_arrayant)
            out = numel(h_qd_arrayant.Pazimuth_grid);
        end
        function out = get.no_el(h_qd_arrayant)
            out = numel(h_qd_arrayant.Pelevation_grid);
        end
        
        % Set functions
        function set.name(h_qd_arrayant,value)
            if ~( ischar(value) )
                error('QuaDRiGa:wrongNumberOfInputs:wrongInputValue','??? "name" must be a string.')
            end
            h_qd_arrayant.name = value;
        end
        
        function set.no_elements(h_qd_arrayant,value)
            if ~( all(size(value) == [1 1]) && isnumeric(value) ...
                    && isreal(value) && mod(value,1)==0 && value > 0 )
                error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "no_elements" must be integer and > 0')
            end
            value = value;
            
            if h_qd_arrayant.no_elements > value
                h_qd_arrayant.Pelement_position = h_qd_arrayant.Pelement_position(:,1:value);
                h_qd_arrayant.PFa = h_qd_arrayant.PFa(:,:,1:value);
                h_qd_arrayant.PFb = h_qd_arrayant.PFb(:,:,1:value);
                                
                ne = h_qd_arrayant.no_elements-value;
                nc = size(h_qd_arrayant.Pcoupling);
                h_qd_arrayant.Pcoupling = h_qd_arrayant.Pcoupling( 1:nc(1)-ne , 1:max(nc(2)-ne,1) );
                
            elseif h_qd_arrayant.no_elements < value
                ne = value-h_qd_arrayant.no_elements;
                
                h_qd_arrayant.Pelement_position = [ h_qd_arrayant.Pelement_position,...
                    h_qd_arrayant.Pelement_position(:,ones( 1,ne )) ];
                
                h_qd_arrayant.PFa = cat( 3, h_qd_arrayant.PFa ,...
                    h_qd_arrayant.PFa(:,:,ones( 1,ne )));
                
                h_qd_arrayant.PFb = cat( 3, h_qd_arrayant.PFb ,...
                    h_qd_arrayant.PFb(:,:,ones( 1,ne )));
                
                nc = size(h_qd_arrayant.Pcoupling);
                C = zeros( nc(1)+ne , nc(2)+ne);
                for n = 1:ne
                   C( nc(1)+n,nc(2)+n ) = 1; 
                end
                C( 1:nc(1) , 1:nc(2) ) = h_qd_arrayant.Pcoupling;
                h_qd_arrayant.Pcoupling = C;
            end
            
            h_qd_arrayant.Pno_elements = value;
        end
        
        function set.elevation_grid(h_qd_arrayant,value)
            if ~( any( size(value) == 1 ) && isnumeric(value) && isreal(value) &&...
                    max(value)<=pi/2+1e-7 && min(value)>=-pi/2-1e-7 )
                error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "elevation_grid" must be a vector containing values between -pi/2 and pi/2')
            end
                        
            val_old = size(h_qd_arrayant.Fa,1);
            val_new = numel(value);
            if val_old > val_new
                h_qd_arrayant.PFa = h_qd_arrayant.PFa( 1:val_new ,: , : );
                h_qd_arrayant.PFb = h_qd_arrayant.PFb( 1:val_new ,: , : );
                
            elseif val_old < val_new
                a = size( h_qd_arrayant.Fa );
                if numel(a) == 2
                    a(3) = 1;
                end
                b = val_new-val_old;
                
                h_qd_arrayant.PFb = cat( 1, h_qd_arrayant.PFb ,...
                    ones( b ,a(2),a(3))  );
                
                h_qd_arrayant.PFa = cat( 1, h_qd_arrayant.PFa ,...
                    ones( b ,a(2),a(3))  );
            end
            
            if size(value,1) ~= 1
                h_qd_arrayant.Pelevation_grid = value';
            else
                h_qd_arrayant.Pelevation_grid = value;
            end
        end
        
        function set.azimuth_grid(h_qd_arrayant,value)
            if ~( any( size(value) == 1 ) && isnumeric(value) && isreal(value) &&...
                     max(value)<=pi+1e-7 && min(value)>=-pi-1e-7 )
                error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "azimuth_grid" must be a vector containing values between -pi and pi')
            end
            
            val_old = size(h_qd_arrayant.Fa,2);
            val_new = numel(value);
            if val_old > val_new
                h_qd_arrayant.PFb = h_qd_arrayant.PFb( : , 1:val_new , : );
                h_qd_arrayant.PFa = h_qd_arrayant.PFa( : , 1:val_new  , : );
                
            elseif val_old < val_new
                a = size( h_qd_arrayant.Fa );
                if numel(a) == 2
                    a(3) = 1;
                end
                b = val_new-val_old;
                
                h_qd_arrayant.PFb = cat( 2, h_qd_arrayant.PFb ,...
                    ones( a(1) , b , a(3))  );
                
                h_qd_arrayant.PFa = cat( 2, h_qd_arrayant.PFa ,...
                    ones( a(1) , b , a(3))  );
            end
            
            if size(value,1) ~= 1
                h_qd_arrayant.Pazimuth_grid = value';
            else
                h_qd_arrayant.Pazimuth_grid = value;
            end
        end
        
        function set.element_position(h_qd_arrayant,value)
            if ~( isnumeric(value) && isreal(value) )
                error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "element_position" must consist of real numbers')
            elseif ~all( size(value,1) == 3 )
                error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "element_position" must have 3 rows')
            end
            if size(value,2) ~= h_qd_arrayant.no_elements
                h_qd_arrayant.no_elements = size(value,2);
            end
            h_qd_arrayant.Pelement_position = value;
        end
        
        function set.Fa(h_qd_arrayant,value)
            a = numel( h_qd_arrayant.Pelevation_grid );
            b = numel( h_qd_arrayant.Pazimuth_grid );
            
            if h_qd_arrayant.Pno_elements == 1
                dims = [ a , b ];
            else
                dims = [ a , b , h_qd_arrayant.Pno_elements];
            end
            
            if ~( isnumeric(value) )
                error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "Fa" must be numeric.')
            elseif ~( numel(size(value)) == numel(dims) && all( size(value) == dims ) )
                error('QuaDRiGa:qd_arrayant:wrongInputValue',['??? "Fa" must be of size [',num2str(a),'x',num2str(b),...
                    'x',num2str(h_qd_arrayant.Pno_elements),'].'])
            end
            h_qd_arrayant.PFa = value;
        end
        
        function set.Fb(h_qd_arrayant,value)
            a = numel( h_qd_arrayant.Pelevation_grid );
            b = numel( h_qd_arrayant.Pazimuth_grid );
            
            if h_qd_arrayant.no_elements == 1
                dims = [ a , b ];
            else
                dims = [ a , b , h_qd_arrayant.Pno_elements];
            end
            
            if ~( isnumeric(value) )
                error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "Fb" must be numeric.')
            elseif ~( numel( size(value) ) == numel( dims ) && all( size(value) == dims ) )
                error('QuaDRiGa:qd_arrayant:wrongInputValue',['??? "Fb" must be of size [',num2str(a),'x',num2str(b),...
                    'x',num2str(h_qd_arrayant.Pno_elements),'].'])
            end
            h_qd_arrayant.PFb = value;
        end
        
        function set.coupling(h_qd_arrayant,value)
            if ~( isnumeric(value) && size(value,1) == h_qd_arrayant.Pno_elements && ...
                    size(value,2) >= 1 )
                error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "coupling" must be a matrix with rows equal to elements and columns equal to ports')
            end
            h_qd_arrayant.Pcoupling = value;
        end
    end

    methods(Static)
        function types = supported_types
            types =  {'omni', 'dipole', 'short-dipole', 'half-wave-dipole',...
                'custom', '3gpp-macro','3gpp-3d','3gpp-mmw',...
                'parametric','rhcp-dipole', 'lhcp-dipole', 'lhcp-rhcp-dipole', ...
                'xpol', 'ula2', 'ula4', 'ula8', 'patch','multi'};
        end
        [ h_qd_arrayant, par ] = generate( array_type, Ain, Bin, Cin, Din, Ein, Fin, Gin, Hin, Iin, Jin );
        h_qd_arrayant = import_pattern( fVi, fHi )
    end
end
