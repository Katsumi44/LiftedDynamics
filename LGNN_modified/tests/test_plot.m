clc;
close all;

%% data loading
load_data = true;
if load_data
    clear;
    data = cell(5,3);
    for i_obj = 1:5
        sweep_freq_file = sprintf('Object%d_sweep_freq.csv',i_obj);
        sweep_magnitude_file = sprintf('Object%d_sweep_magnitude.csv',i_obj);
        sweep_scale_file = sprintf('Object%d_sweep_scale.csv',i_obj);
        if exist(sweep_freq_file,'file') == 2
            data{i_obj,1} = readtable(sweep_freq_file);
            data{i_obj,2} = readtable(sweep_magnitude_file);
            data{i_obj,3} = readtable(sweep_scale_file);
        else
            continue;
        end
    end
end

%% parameters
set_data_num = 5*100;
set_num = 50;
DIY_font_size = 30;
DIY_font_name = 'Times New Roman';
DIY_line_width = 2;
DIY_marker = 'o';
analyse_sweep_freq = true;
analyse_sweep_magni = true;
analyse_sweep_scale = true;
i_obj = 5;
if i_obj == 5
    N = 8;
else
    N = 5;
end

%% frequency sweep analysis
i_eval = 1;
if analyse_sweep_freq
    if isempty(data{i_obj,i_eval})==0
        data_temp = data{i_obj,i_eval};

        % position error - freq
        figure(1+100*i_eval);
        for i_node = 1:N
            x_name = sprintf('Position_Error_Node%d_Dim0',i_node-1);
            x_data = data_temp.(x_name);
            y_name = sprintf('Position_Error_Node%d_Dim1',i_node-1);
            y_data = data_temp.(y_name);
            z_name = sprintf('Position_Error_Node%d_Dim2',i_node-1);
            z_data = data_temp.(z_name);
            error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
            t = data_temp.Time;
            freq = data_temp.Frequency;
            freq_unique = unique(freq,'rows','stable');
            error_mean = zeros(set_num,1);
            for i_set = 1:set_num
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                error_mean(i_set) = mean(error(indices,1));
            end
            plot(freq_unique,error_mean,LineWidth=DIY_line_width,Marker=DIY_marker);
            hold on;
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        legend('Node 1','Node 2','Node 3','Node 4','Node 5');
        xlabel('Frequency (Hz)');
        ylabel('Mean Position Error (m)');
        title('Frequency Sweep');

        % velocity error - freq
        figure(2+100*i_eval);
        for i_node = 1:N
            x_name = sprintf('Velocity_Error_Node%d_Dim0',i_node-1);
            x_data = data_temp.(x_name);
            y_name = sprintf('Velocity_Error_Node%d_Dim1',i_node-1);
            y_data = data_temp.(y_name);
            z_name = sprintf('Velocity_Error_Node%d_Dim2',i_node-1);
            z_data = data_temp.(z_name);
            error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
            t = data_temp.Time;
            freq = data_temp.Frequency;
            freq_unique = unique(freq,'rows','stable');
            error_mean = zeros(set_num,1);
            for i_set = 1:set_num
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                error_mean(i_set) = mean(error(indices,1));
            end
            plot(freq_unique,error_mean,LineWidth=DIY_line_width,Marker=DIY_marker);
            hold on;
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        legend('Node 1','Node 2','Node 3','Node 4','Node 5');
        xlabel('Frequency (Hz)');
        ylabel('Mean Velocity Error (m/s)');
        title('Frequency Sweep');

        % position error - time
        figure(3+100*i_eval);
        i_node = 2;
        freq_thresh = 2.5;
        x_name = sprintf('Position_Error_Node%d_Dim0',i_node-1);
        x_data = data_temp.(x_name);
        y_name = sprintf('Position_Error_Node%d_Dim1',i_node-1);
        y_data = data_temp.(y_name);
        z_name = sprintf('Position_Error_Node%d_Dim2',i_node-1);
        z_data = data_temp.(z_name);
        error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
        t = data_temp.Time;
        freq = data_temp.Frequency;
        freq_unique = unique(freq,'rows','stable');
        for i_set = 1:set_num
            if freq_unique(i_set,1) < freq_thresh
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                t_set = t(indices,1);
                error_set = error(indices,1);
                plot(t_set,error_set,LineWidth=DIY_line_width);
                hold on;
            end
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        xlabel('Time (s)');
        ylabel('Position Error (m)');
        title(sprintf('Position Error Over Time for Node %d (Frequency < %.2f Hz)',i_node,freq_thresh));

        % velocity error - time
        figure(4+100*i_eval);
        i_node = 2;
        freq_thresh = 2.5;
        x_name = sprintf('Velocity_Error_Node%d_Dim0',i_node-1);
        x_data = data_temp.(x_name);
        y_name = sprintf('Velocity_Error_Node%d_Dim1',i_node-1);
        y_data = data_temp.(y_name);
        z_name = sprintf('Velocity_Error_Node%d_Dim2',i_node-1);
        z_data = data_temp.(z_name);
        error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
        t = data_temp.Time;
        freq = data_temp.Frequency;
        freq_unique = unique(freq,'rows','stable');
        for i_set = 1:set_num
            if freq_unique(i_set,1) < freq_thresh
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                t_set = t(indices,1);
                error_set = error(indices,1);
                plot(t_set,error_set,LineWidth=DIY_line_width);
                hold on;
            end
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        xlabel('Time (s)');
        ylabel('Velocity Error (m/s)');
        title(sprintf('Velocity Error Over Time for Node %d (Frequency < %.2f Hz)',i_node,freq_thresh));

        % position - time
        i_node = 2;
        freq_thresh = 2.5;
        fig_num = 2;
        x_pred_name = sprintf('Position_Predict_Node%d_Dim0',i_node-1);
        x_pred_data = data_temp.(x_pred_name);
        y_pred_name = sprintf('Position_Predict_Node%d_Dim1',i_node-1);
        y_pred_data = data_temp.(y_pred_name);
        x_truth_name = sprintf('Position_Truth_Node%d_Dim0',i_node-1);
        x_truth_data = data_temp.(x_truth_name);
        y_truth_name = sprintf('Position_Truth_Node%d_Dim1',i_node-1);
        y_truth_data = data_temp.(y_truth_name);
        t = data_temp.Time;
        freq = data_temp.Frequency;
        freq_unique = unique(freq,'rows','stable');
        i_set_thresh = find(freq_unique>freq_thresh);
        for i_fig = 1:fig_num
            i_set = i_set_thresh(i_fig);
            indices = 1:set_data_num;
            indices = indices + (i_set-1) * set_data_num;
            t_set = t(indices,1);
            x_pred_set = x_pred_data(indices,1);
            y_pred_set = y_pred_data(indices,1);
            x_truth_set = x_truth_data(indices,1);
            y_truth_set = y_truth_data(indices,1);

            figure(4+i_fig+100*i_eval);

            subplot(2,1,1);
            plot(t_set,x_pred_set,LineWidth=DIY_line_width,Color='blue');
            hold on;
            plot(t_set,x_truth_set,LineWidth=DIY_line_width,Color='red');
            hold off;
            set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
            legend('Prediction','Ground Truth');
            xlabel('Time (s)');
            ylabel('Position (m)');
            title(sprintf('X-Axis Position Over Time for Node %d (Frequency = %.2f Hz)',i_node,freq_unique(i_set)));

            subplot(2,1,2);
            plot(t_set,y_pred_set,LineWidth=DIY_line_width,Color='blue');
            hold on;
            plot(t_set,y_truth_set,LineWidth=DIY_line_width,Color='red');
            hold off;
            set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
            legend('Prediction','Ground Truth');
            xlabel('Time (s)');
            ylabel('Position (m)');
            title(sprintf('Y-Axis Position Over Time for Node %d (Frequency = %.2f Hz)',i_node,freq_unique(i_set)));
        end
    end
end

%% magnitude sweep analysis
i_eval = 2;
if analyse_sweep_magni
    if isempty(data{i_obj,i_eval})==0
        data_temp = data{i_obj,i_eval};

        % position error - magni
        figure(1+100*i_eval);
        for i_node = 1:N
            x_name = sprintf('Position_Error_Node%d_Dim0',i_node-1);
            x_data = data_temp.(x_name);
            y_name = sprintf('Position_Error_Node%d_Dim1',i_node-1);
            y_data = data_temp.(y_name);
            z_name = sprintf('Position_Error_Node%d_Dim2',i_node-1);
            z_data = data_temp.(z_name);
            error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
            t = data_temp.Time;
            magni = data_temp.Magnitude;
            magni_unique = unique(magni,'rows','stable');
            error_mean = zeros(set_num,1);
            for i_set = 1:set_num
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                error_mean(i_set) = mean(error(indices,1));
            end
            plot(magni_unique,error_mean,LineWidth=DIY_line_width,Marker=DIY_marker);
            hold on;
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        legend('Node 1','Node 2','Node 3','Node 4','Node 5');
        xlabel('Magnitude (m)');
        ylabel('Mean Position Error (m)');
        title('Magnitude Sweep');

        % velocity error - magni
        figure(2+100*i_eval);
        for i_node = 1:N
            x_name = sprintf('Velocity_Error_Node%d_Dim0',i_node-1);
            x_data = data_temp.(x_name);
            y_name = sprintf('Velocity_Error_Node%d_Dim1',i_node-1);
            y_data = data_temp.(y_name);
            z_name = sprintf('Velocity_Error_Node%d_Dim2',i_node-1);
            z_data = data_temp.(z_name);
            error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
            t = data_temp.Time;
            magni = data_temp.Magnitude;
            magni_unique = unique(magni,'rows','stable');
            error_mean = zeros(set_num,1);
            for i_set = 1:set_num
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                error_mean(i_set) = mean(error(indices,1));
            end
            plot(magni_unique,error_mean,LineWidth=DIY_line_width,Marker=DIY_marker);
            hold on;
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        legend('Node 1','Node 2','Node 3','Node 4','Node 5');
        xlabel('Magnitude (m)');
        ylabel('Mean Velocity Error (m/s)');
        title('Magnitude Sweep');

        % position error - time
        figure(3+100*i_eval);
        i_node = 2;
        magni_thresh = 0.28;
        x_name = sprintf('Position_Error_Node%d_Dim0',i_node-1);
        x_data = data_temp.(x_name);
        y_name = sprintf('Position_Error_Node%d_Dim1',i_node-1);
        y_data = data_temp.(y_name);
        z_name = sprintf('Position_Error_Node%d_Dim2',i_node-1);
        z_data = data_temp.(z_name);
        error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
        t = data_temp.Time;
        magni = data_temp.Magnitude;
        magni_unique = unique(magni,'rows','stable');
        for i_set = 1:set_num
            if magni_unique(i_set,1) < magni_thresh
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                t_set = t(indices,1);
                error_set = error(indices,1);
                plot(t_set,error_set,LineWidth=DIY_line_width);
                hold on;
            end
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        xlabel('Time (s)');
        ylabel('Position Error (m)');
        title(sprintf('Position Error Over Time for Node %d (Magnitude < %.3f m)',i_node,magni_thresh));

        % velocity error - time
        figure(4+100*i_eval);
        i_node = 2;
        magni_thresh = 0.28;
        x_name = sprintf('Velocity_Error_Node%d_Dim0',i_node-1);
        x_data = data_temp.(x_name);
        y_name = sprintf('Velocity_Error_Node%d_Dim1',i_node-1);
        y_data = data_temp.(y_name);
        z_name = sprintf('Velocity_Error_Node%d_Dim2',i_node-1);
        z_data = data_temp.(z_name);
        error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
        t = data_temp.Time;
        magni = data_temp.Magnitude;
        magni_unique = unique(magni,'rows','stable');
        for i_set = 1:set_num
            if magni_unique(i_set,1) < magni_thresh
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                t_set = t(indices,1);
                error_set = error(indices,1);
                plot(t_set,error_set,LineWidth=DIY_line_width);
                hold on;
            end
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        xlabel('Time (s)');
        ylabel('Velocity Error (m/s)');
        title(sprintf('Velocity Error Over Time for Node %d (Magnitude < %.3f m)',i_node,magni_thresh));

        % position - time
        i_node = 2;
        magni_thresh = 0.28;
        fig_num = 2;
        x_pred_name = sprintf('Position_Predict_Node%d_Dim0',i_node-1);
        x_pred_data = data_temp.(x_pred_name);
        y_pred_name = sprintf('Position_Predict_Node%d_Dim1',i_node-1);
        y_pred_data = data_temp.(y_pred_name);
        x_truth_name = sprintf('Position_Truth_Node%d_Dim0',i_node-1);
        x_truth_data = data_temp.(x_truth_name);
        y_truth_name = sprintf('Position_Truth_Node%d_Dim1',i_node-1);
        y_truth_data = data_temp.(y_truth_name);
        t = data_temp.Time;
        magni = data_temp.Magnitude;
        magni_unique = unique(magni,'rows','stable');
        i_set_thresh = find(magni_unique>magni_thresh);
        for i_fig = 1:fig_num
            i_set = i_set_thresh(i_fig);
            indices = 1:set_data_num;
            indices = indices + (i_set-1) * set_data_num;
            t_set = t(indices,1);
            x_pred_set = x_pred_data(indices,1);
            y_pred_set = y_pred_data(indices,1);
            x_truth_set = x_truth_data(indices,1);
            y_truth_set = y_truth_data(indices,1);

            figure(4+i_fig+100*i_eval);

            subplot(2,1,1);
            plot(t_set,x_pred_set,LineWidth=DIY_line_width,Color='blue');
            hold on;
            plot(t_set,x_truth_set,LineWidth=DIY_line_width,Color='red');
            hold off;
            set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
            legend('Prediction','Ground Truth');
            xlabel('Time (s)');
            ylabel('Position (m)');
            title(sprintf('X-Axis Position Over Time for Node %d (Magnitude = %.3f m)',i_node,magni_unique(i_set)));

            subplot(2,1,2);
            plot(t_set,y_pred_set,LineWidth=DIY_line_width,Color='blue');
            hold on;
            plot(t_set,y_truth_set,LineWidth=DIY_line_width,Color='red');
            hold off;
            set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
            legend('Prediction','Ground Truth');
            xlabel('Time (s)');
            ylabel('Position (m)');
            title(sprintf('Y-Axis Position Over Time for Node %d (Magnitude = %.3f m)',i_node,magni_unique(i_set)));
        end
    end
end

%% scale sweep analysis
i_eval = 3;
if analyse_sweep_scale
    if isempty(data{i_obj,i_eval})==0
        data_temp = data{i_obj,i_eval};

        % position error - magni
        figure(1+100*i_eval);
        for i_node = 1:N
            x_name = sprintf('Position_Error_Node%d_Dim0',i_node-1);
            x_data = data_temp.(x_name);
            y_name = sprintf('Position_Error_Node%d_Dim1',i_node-1);
            y_data = data_temp.(y_name);
            z_name = sprintf('Position_Error_Node%d_Dim2',i_node-1);
            z_data = data_temp.(z_name);
            error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
            t = data_temp.Time;
            magni = data_temp.Magnitude;
            magni_unique = unique(magni,'rows','stable');
            error_mean = zeros(set_num,1);
            for i_set = 1:set_num
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                error_mean(i_set) = mean(error(indices,1));
            end
            plot(magni_unique,error_mean,LineWidth=DIY_line_width,Marker=DIY_marker);
            hold on;
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        legend('Node 1','Node 2','Node 3','Node 4','Node 5');
        xlabel('Magnitude (m)');
        ylabel('Mean Position Error (m)');
        title('Magnitude Sweep');

        % velocity error - magni
        figure(2+100*i_eval);
        for i_node = 1:N
            x_name = sprintf('Velocity_Error_Node%d_Dim0',i_node-1);
            x_data = data_temp.(x_name);
            y_name = sprintf('Velocity_Error_Node%d_Dim1',i_node-1);
            y_data = data_temp.(y_name);
            z_name = sprintf('Velocity_Error_Node%d_Dim2',i_node-1);
            z_data = data_temp.(z_name);
            error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
            t = data_temp.Time;
            magni = data_temp.Magnitude;
            magni_unique = unique(magni,'rows','stable');
            error_mean = zeros(set_num,1);
            for i_set = 1:set_num
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                error_mean(i_set) = mean(error(indices,1));
            end
            plot(magni_unique,error_mean,LineWidth=DIY_line_width,Marker=DIY_marker);
            hold on;
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        legend('Node 1','Node 2','Node 3','Node 4','Node 5');
        xlabel('Magnitude (m)');
        ylabel('Mean Velocity Error (m/s)');
        title('Magnitude Sweep');

        % position error - time
        figure(3+100*i_eval);
        i_node = 2;
        magni_thresh = 0.28;
        x_name = sprintf('Position_Error_Node%d_Dim0',i_node-1);
        x_data = data_temp.(x_name);
        y_name = sprintf('Position_Error_Node%d_Dim1',i_node-1);
        y_data = data_temp.(y_name);
        z_name = sprintf('Position_Error_Node%d_Dim2',i_node-1);
        z_data = data_temp.(z_name);
        error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
        t = data_temp.Time;
        magni = data_temp.Magnitude;
        magni_unique = unique(magni,'rows','stable');
        for i_set = 1:set_num
            if magni_unique(i_set,1) < magni_thresh
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                t_set = t(indices,1);
                error_set = error(indices,1);
                plot(t_set,error_set,LineWidth=DIY_line_width);
                hold on;
            end
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        xlabel('Time (s)');
        ylabel('Position Error (m)');
        title(sprintf('Position Error Over Time for Node %d (Magnitude < %.3f m)',i_node,magni_thresh));

        % velocity error - time
        figure(4+100*i_eval);
        i_node = 2;
        magni_thresh = 0.28;
        x_name = sprintf('Velocity_Error_Node%d_Dim0',i_node-1);
        x_data = data_temp.(x_name);
        y_name = sprintf('Velocity_Error_Node%d_Dim1',i_node-1);
        y_data = data_temp.(y_name);
        z_name = sprintf('Velocity_Error_Node%d_Dim2',i_node-1);
        z_data = data_temp.(z_name);
        error = (x_data.^2 + y_data.^2 + z_data.^2) .^ (1/2);
        t = data_temp.Time;
        magni = data_temp.Magnitude;
        magni_unique = unique(magni,'rows','stable');
        for i_set = 1:set_num
            if magni_unique(i_set,1) < magni_thresh
                indices = 1:set_data_num;
                indices = indices + (i_set-1) * set_data_num;
                t_set = t(indices,1);
                error_set = error(indices,1);
                plot(t_set,error_set,LineWidth=DIY_line_width);
                hold on;
            end
        end
        set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
        xlabel('Time (s)');
        ylabel('Velocity Error (m/s)');
        title(sprintf('Velocity Error Over Time for Node %d (Magnitude < %.3f m)',i_node,magni_thresh));

        % position - time
        i_node = 2;
        magni_thresh = 0.28;
        fig_num = 2;
        x_pred_name = sprintf('Position_Predict_Node%d_Dim0',i_node-1);
        x_pred_data = data_temp.(x_pred_name);
        y_pred_name = sprintf('Position_Predict_Node%d_Dim1',i_node-1);
        y_pred_data = data_temp.(y_pred_name);
        x_truth_name = sprintf('Position_Truth_Node%d_Dim0',i_node-1);
        x_truth_data = data_temp.(x_truth_name);
        y_truth_name = sprintf('Position_Truth_Node%d_Dim1',i_node-1);
        y_truth_data = data_temp.(y_truth_name);
        t = data_temp.Time;
        magni = data_temp.Magnitude;
        magni_unique = unique(magni,'rows','stable');
        i_set_thresh = find(magni_unique>magni_thresh);
        for i_fig = 1:fig_num
            i_set = i_set_thresh(i_fig);
            indices = 1:set_data_num;
            indices = indices + (i_set-1) * set_data_num;
            t_set = t(indices,1);
            x_pred_set = x_pred_data(indices,1);
            y_pred_set = y_pred_data(indices,1);
            x_truth_set = x_truth_data(indices,1);
            y_truth_set = y_truth_data(indices,1);

            figure(4+i_fig+100*i_eval);

            subplot(2,1,1);
            plot(t_set,x_pred_set,LineWidth=DIY_line_width,Color='blue');
            hold on;
            plot(t_set,x_truth_set,LineWidth=DIY_line_width,Color='red');
            hold off;
            set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
            legend('Prediction','Ground Truth');
            xlabel('Time (s)');
            ylabel('Position (m)');
            title(sprintf('X-Axis Position Over Time for Node %d (Magnitude = %.3f m)',i_node,magni_unique(i_set)));

            subplot(2,1,2);
            plot(t_set,y_pred_set,LineWidth=DIY_line_width,Color='blue');
            hold on;
            plot(t_set,y_truth_set,LineWidth=DIY_line_width,Color='red');
            hold off;
            set(gca,fontname=DIY_font_name,fontsize=DIY_font_size);
            legend('Prediction','Ground Truth');
            xlabel('Time (s)');
            ylabel('Position (m)');
            title(sprintf('Y-Axis Position Over Time for Node %d (Magnitude = %.3f m)',i_node,magni_unique(i_set)));
        end
    end
end

