/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.spark.mlib.ioclass;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.ReflectionUtils;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

/**
 * Created by qinzhiheng on 2017/7/17.
 *
 * Class used to solve GBK output data for spark
 *
 */
public class MyOutputFormat<K, V> extends FileOutputFormat<K, V> {
    private static String SEPERATOR = "mapreduce.output.textoutputformat.separator";
    private static class LineRecordWriter<K, V>
            extends RecordWriter<K, V> {
        private static final String utf8 = "gbk";
        private static final byte[] newline;
        static {
            try {
                newline = "\n".getBytes(utf8);
            } catch (UnsupportedEncodingException uee) {
                throw new IllegalArgumentException("can't find " + utf8 + " encoding");
            }
        }

        private DataOutputStream out;
        private final byte[] keyValueSeparator;

        private LineRecordWriter(DataOutputStream out, String keyValueSeparator) {
            this.out = out;
            try {
                this.keyValueSeparator = keyValueSeparator.getBytes(utf8);
            } catch (UnsupportedEncodingException uee) {
                throw new IllegalArgumentException("can't find " + utf8 + " encoding");
            }
        }

        public LineRecordWriter(DataOutputStream out) {
            this(out, "\t");
        }

        /**
         * Write the object to the byte stream, handling Text as a special
         * case.
         * @param o the object to print
         * @throws IOException if the write throws, we pass it on
         */
    /*private void writeObject(Object o) throws IOException {
      if (o instanceof Text) {
        Text to = (Text) o;
        out.write(to.getBytes(), 0, to.getLength());
      } else {
        out.write(o.toString().getBytes(utf8));
      }
    }*/
        private void writeObject(Object o) throws IOException {
            out.write(o.toString().getBytes(utf8));
        }
        public synchronized void write(K key, V value)
                throws IOException {

            boolean nullKey = key == null || key instanceof NullWritable;
            boolean nullValue = value == null || value instanceof NullWritable;
            if (nullKey && nullValue) {
                return;
            }
            if (!nullKey) {
                writeObject(key);
            }
            if (!(nullKey || nullValue)) {
                out.write(keyValueSeparator);
            }
            if (!nullValue) {
                writeObject(value);
            }
            out.write(newline);
        }

        public synchronized void close(TaskAttemptContext context) throws IOException {
            out.close();
        }
    }

    public RecordWriter<K, V>
    getRecordWriter(TaskAttemptContext job) throws IOException, InterruptedException {
        Configuration conf = job.getConfiguration();
        boolean isCompressed = getCompressOutput(job);
        String keyValueSeparator= conf.get(SEPERATOR, "\t");
        CompressionCodec codec = null;
        String extension = "";
        if (isCompressed) {
            Class<? extends CompressionCodec> codecClass =
                    getOutputCompressorClass(job, GzipCodec.class);
            codec = (CompressionCodec) ReflectionUtils.newInstance(codecClass, conf);
            extension = codec.getDefaultExtension();
        }
        Path file = getDefaultWorkFile(job, extension);
        FileSystem fs = file.getFileSystem(conf);
        if (!isCompressed) {
            FSDataOutputStream fileOut = fs.create(file, false);
            return new LineRecordWriter<K, V>(fileOut, keyValueSeparator);
        } else {
            FSDataOutputStream fileOut = fs.create(file, false);
            return new LineRecordWriter<K, V>(new DataOutputStream
                    (codec.createOutputStream(fileOut)),
                    keyValueSeparator);
        }
    }
}

