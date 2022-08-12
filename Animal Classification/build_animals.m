% Dataset Preparation
%% clear
    clear all;
%% Dataset Preparation
files = dir(fullfile('animals','*.jpg'));  
    for j = 1:3000
        every_animal= imread(fullfile('animals',files(j).name));
        every_animal =imresize(every_animal,[40 40]);
        every_animal = rgb2gray(every_animal);
        every_animal = reshape(every_animal,[],1);
        X(:,j)=every_animal;  
    end
    X=im2double(X);

    Cat =[1,0,0];
    Dog =[0,1,0];
    Panda =[0,0,1];
    y =[];
    for i =1:1000
        y=[y Cat'];
    end
    for i =1001:2000
        y=[y Dog'];
    end
    for i =2001:3000
        y=[y Panda'];
    end
 %% save data
 save('data.mat', 'X', 'y');
                
        
 
    
    
    

