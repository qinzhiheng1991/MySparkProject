package com.spark.mlib.ioclass;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.LineReader;

import java.io.IOException;
import java.util.StringTokenizer;

/**
 * Created by qinzhiheng on 2017/7/17.
 *
 * Class used to solve GBK input data for spark
 *
 */

public class MyInputFormat extends FileInputFormat<LongWritable,Text> {

    public MyInputFormat(){}

    @Override
    public RecordReader<LongWritable,Text> createRecordReader(
            InputSplit split, TaskAttemptContext context) throws IOException,
            InterruptedException {
        // TODO Auto-generated method stub
        return new MyRecordReader();
    }
    @Override
    protected boolean isSplitable(JobContext context, Path filename)
    {
        return false;
    }
    public static class MyRecordReader extends RecordReader<LongWritable,Text>{

        public LineReader in;
        public LongWritable key;
        public Text value;
        public StringTokenizer token = null;
        public Text line;

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context)
                throws IOException, InterruptedException {
            // TODO Auto-generated method stub
            FileSplit fileSplit = (FileSplit)split;
            Configuration job = context.getConfiguration();
            Path file = fileSplit.getPath();
            FileSystem fs = file.getFileSystem(job);

            FSDataInputStream filein = fs.open(file);
            in = new LineReader(filein, job);

            key = new LongWritable();
            value = new Text();
            line = new Text();
        }

        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException {

            int linesize = in.readLine(line);
            if(linesize==0)
            {
                return false;
            }
            String val = new String(line.getBytes(),0,line.getLength(),"gbk");
            value = new Text(val);
            return true;
        }

        @Override
        public LongWritable getCurrentKey() throws IOException,
                InterruptedException {
            // TODO Auto-generated method stub
            return key;
        }

        @Override
        public Text getCurrentValue() throws IOException,
                InterruptedException {
            // TODO Auto-generated method stub
            return value;
        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            // TODO Auto-generated method stub
            return 0;
        }

        @Override
        public void close() throws IOException {
            // TODO Auto-generated method stub

        }

    }

}