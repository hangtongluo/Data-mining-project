clc
clear all
close all
% for i=1:2
%     if i == 1
%         disp('Step1:ѡ��Ŀ������1');
%         quyushengzhang(i); close;
%         quyushengzhang(i); close;
%     else
%         disp('Step2:ѡ��Ŀ������2');
%         quyushengzhang(i); close;
%     end
%     
% end
I1 = imread('1.jpg');
I2 = imread('2.jpg');
figure('Name','�ָ���Ч��ͼ');
subplot(121);imshow(I1);title('Ŀ������1');
subplot(122);imshow(I2);title('Ŀ������2');
saveas(gcf,'Ŀ��������ȡЧ��ͼ.jpg')

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
% figure('Name','��ԭ���Ч��ͼ');
% imshow(uint8(I4));title('��ԭ���Ч��ͼ');
% imwrite(uint8(I4),'��ԭ���Ч��ͼ.jpg');

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
figure('Name','��ԭ���Ч��ͼ');
imshow(uint8(I4));title('��ԭ���Ч��ͼ');
imwrite(uint8(I4),'��ԭ���Ч��ͼ.jpg');

[m,n] = size(I1);
count1 = 0;
for i = 1:m
    for j= 1:n
        if I1(i,j) > 0
            count1 = count1+1;
        end
    end
end
%����1ռ����ͼ�������
p1 = count1/(m*n)

count2=0;
for i = 1:m
    for j= 1:n
        if I2(i,j) > 0
            count2 = count2+1;
        end
    end
end
%����2ռ����ͼ�������
p2 = count2/(m*n)

%����1ռ����2�������
P_end = count2/count1







