clc
clear all
close all
% for i=1:2
%     if i == 1
%         disp('Step1:选择目标区域1');
%         quyushengzhang(i); close;
%         quyushengzhang(i); close;
%     else
%         disp('Step2:选择目标区域2');
%         quyushengzhang(i); close;
%     end
%     
% end
I1 = imread('1.jpg');
I2 = imread('2.jpg');
figure('Name','分割后的效果图');
subplot(121);imshow(I1);title('目标区域1');
subplot(122);imshow(I2);title('目标区域2');
saveas(gcf,'目标区域提取效果图.jpg')

[m,n] = size(I1);
I3 = [];
for i = 1:m
    for j= 1:n
        if I1(i,j) == 0
           I3(i,j) = 255;
        else
            I3(i,j) = 0;
        end
    end
end

I1 = double(I1);
I2 = double(I2);
I3 = double(I3);
I4 = I3+I2;
% figure('Name','还原后的效果图');
% imshow(uint8(I4));title('还原后的效果图');
% imwrite(uint8(I4),'还原后的效果图.jpg');

[m,n] = size(I4);
for i = 1:m
    for j= 1:n
        if I4(i,j) == 0
           I4(i,j) = 255;
        else
            I4(i,j) = 0;
        end
    end
end
figure('Name','还原后的效果图');
imshow(uint8(I4));title('还原后的效果图');
imwrite(uint8(I4),'还原后的效果图.jpg');

[m,n] = size(I1);
count1 = 0;
for i = 1:m
    for j= 1:n
        if I1(i,j) > 0
            count1 = count1+1;
        end
    end
end
%区域1占整幅图的面积比
p1 = count1/(m*n)

count2=0;
for i = 1:m
    for j= 1:n
        if I2(i,j) > 0
            count2 = count2+1;
        end
    end
end
%区域2占整幅图的面积比
p2 = count2/(m*n)

%区域1占区域2的面积比
P_end = count2/count1







