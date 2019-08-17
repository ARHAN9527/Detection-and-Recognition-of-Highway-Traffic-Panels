package com.example.panelapplication;

import com.example.panelapplication.env.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

public class Postprocessing {
    private static final Logger LOGGER = new Logger();
    private float [][]tfoutput;
    private ArrayList<float[]> nmsBoxes = new ArrayList<float[]>();

    public Postprocessing(float [][]tfoutput) {
        this.tfoutput = tfoutput;
        //scores = new float[tfoutput.length];
        //bboxes = new float[tfoutput.length][4];
    }

    private void bboxesClip(float[][] array, int indexA, int indexB) {
        float []rbImg = {0f, 0f, 1f, 1f};

        float []j;
        /*for (int i=0; i < this.tfoutput.length; i++) {
            j = this.tfoutput[i];
            scores[i] = j[0];
            bboxes[i][0] =  j[1] > rbImg[0] ? j[1] : rbImg[0];
            bboxes[i][1] =  j[2] > rbImg[1] ? j[2] : rbImg[1];
            bboxes[i][2] =  j[3] < rbImg[2] ? j[3] : rbImg[2];
            bboxes[i][3] =  j[4] < rbImg[3] ? j[4] : rbImg[3];
        }*/

        array[indexA][1] =  array[indexA][1] > rbImg[0] ? array[indexA][1] : rbImg[0];
        array[indexA][2] =  array[indexA][2] > rbImg[1] ? array[indexA][2] : rbImg[1];
        array[indexA][3] =  array[indexA][3] < rbImg[2] ? array[indexA][3] : rbImg[2];
        array[indexA][4] =  array[indexA][4] < rbImg[3] ? array[indexA][4] : rbImg[3];

        array[indexB][1] =   array[indexB][1] > rbImg[0] ?  array[indexB][1] : rbImg[0];
        array[indexB][2] =   array[indexB][2] > rbImg[1] ?  array[indexB][2] : rbImg[1];
        array[indexB][3] =   array[indexB][3] < rbImg[2] ?  array[indexB][3] : rbImg[2];
        array[indexB][4] =   array[indexB][4] < rbImg[3] ?  array[indexB][4] : rbImg[3];

    }

    private void swap(float[][] array, int indexA, int indexB) {
        float []tmp = array[indexA];
        array[indexA] = array[indexB];
        array[indexB] = tmp;

        bboxesClip(array, indexA, indexB);
    }

    private void sort(float[][] array, int left, int right) {
        if (left >= right)
            return;

        // random pivot
        //int pivotIndex = left + random.nextInt(right - left + 1);

        // middle pivot
        int pivotIndex = (left + right) / 2;
        float pivot = array[pivotIndex][0];
        swap(array, pivotIndex, right);
        int swapIndex = left;
        for (int i = left; i < right; i++)
        {
            if (array[i][0] >= pivot)
            {
                swap(array, i, swapIndex);
                swapIndex++;
            }
        }
        swap(array, swapIndex, right);
        sort(array, left, swapIndex - 1);
        sort(array, swapIndex + 1, right);
    }

    private void bboxesSort(float[][] array) {
        sort(array, 0, array.length-1);
        /*new Thread(new Runnable() {
            public void run() {
                sort(tfoutput, 0, scores.length-1);
            }
        }).start();*/
    }

    private float bboxesIou(float []box1, float []box2) {

        float ymin =  box1[1] > box2[1] ? box1[1] : box2[1];
        float xmin =  box1[2] > box2[2] ? box1[2] : box2[2];
        float ymax =  box1[3] < box2[3] ? box1[3] : box2[3];
        float xmax =  box1[4] < box2[4] ? box1[4] : box2[4];

        float h = (ymax-ymin);
        h = h > 0 ? h : 0;
        float w = (xmax-xmin);
        w = w > 0 ? w : 0;

        float vol = h * w;

        float vol1 = (box1[3] - box1[1]) * (box1[4] - box1[2]);
        float vol2 = (box2[3] - box2[1]) * (box2[4] - box2[2]);

        return vol / (vol1 + vol2 - vol);
    }

    private ArrayList<float[]> bboxesNms(float[][] array) {
        final int topK = 50;
        final int numDetection=5;
        final float nmsThreshhold = 0.5f;
        boolean keepOverlap;
        float iou;
        boolean []keepBox = new boolean[topK];

        Arrays.fill(keepBox, true);

        for (int i=0; i < topK; i++){
            if (keepBox[i] && array[i][0] > 0.5){
                nmsBoxes.add(array[i]);
                if (nmsBoxes.size() == numDetection)
                    break;
                for (int j=i+1; j < topK; j++){
                    if (keepBox[j] == false)
                        continue;
                    iou = bboxesIou(array[i], array[j]);
                    keepOverlap = iou < nmsThreshhold;
                    keepBox[j] = keepOverlap;
                    //keepBox[j] = keepBox[j] && keepOverlap;
                }
            }
        }

        return nmsBoxes;
    }

    public ArrayList<float[]> postDetect() {
        bboxesSort(tfoutput);
        return bboxesNms(tfoutput);
    }

    private int argmax(float[][] array) {
        float maximum = array[0][0];
        int idx = 0;
        for (int i = 2; i < array[0].length; i++)
        {
            if (array[0][i] > maximum && array[0][i] > 0.8)
            {
                maximum = array[0][i];
                idx = i;
            }
        }
        return idx;
    }

    public int postRecognize() {
        return argmax(tfoutput);
    }


    public void printOutput() {
        int index=0;
        for (float []i : this.tfoutput) {
            LOGGER.w("index="+index + " score="+i[0] + " box="+i[1]+","+i[2]+","+i[3]+","+i[4]);
            index++;
        }

    }
}
