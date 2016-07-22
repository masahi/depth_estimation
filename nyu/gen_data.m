load train_scenes.mat
addpath('toolbox')
n_scenes = size(train_scenes,1);

datasetDir = 'data';

all_file_names = fopen('file_names.txt', 'w');
failed_scenes = fopen('failed_scenes.txt', 'w');

for i=1:n_scenes
    sceneName = strtrim(train_scenes(i, :));
    try
      framelist = get_synched_frames(['data/' sceneName ]);
    catch 
      fprintf(failed_scenes  , sceneName);
      continue  
    end   
    
    sceneDir = sprintf('%s/%s', datasetDir, sceneName);    
    savedir = sprintf('training/%s', sceneName);
    mkdir(savedir)
    n_frames = length(framelist);
    
    for j=[1:5:n_frames]
        
        file_name = [savedir '/' int2str(j) '.mat'];
        
        try
          rgb = crop_image(imread([sceneDir '/' framelist(j).rawRgbFilename]));
          raw_depth = swapbytes(imread([sceneDir '/' framelist(j).rawDepthFilename]));
          depth = crop_image(project_depth_map(raw_depth, rgb)); ...
                  
          save(file_name, 'rgb', 'depth')
          fprintf(all_file_names, [file_name '\n']);
                  
        catch
          continue  
        end    
        

    end
end    

fclose(all_file_names);
fclose(failed_scenes);

