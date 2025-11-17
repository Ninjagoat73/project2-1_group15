using System;
using System.Diagnostics;
using Unity.MLAgents;

public class PerformanceLogger
{
    private ProcessStruct mainProcessStruct;
    private DateTime lastStepMeasure;
    private DateTime lastEpisodeMeasure;
    private PerformanceCounter cpuCounter;
    private StatsRecorder recorder;
    private Agent agent;
    private int frequency;

    public PerformanceLogger(StatsRecorder recorder, Process mainProcess, int frequency)
    {
        this.mainProcessStruct = new ProcessStruct(mainProcess);
        this.recorder = recorder;
        this.lastStepMeasure = DateTime.Now;
        this.lastEpisodeMeasure = DateTime.Now;
        this.frequency = frequency;
    }
    private float GetCpuUsage(ProcessStruct processCpuStruct)
    {
        Process process = processCpuStruct.process;
        var now = DateTime.UtcNow;
        var cpuUsed = process.TotalProcessorTime - processCpuStruct.lastCpuTime;
        var elapsed = now - processCpuStruct.lastSampleTime;

        processCpuStruct.lastCpuTime = process.TotalProcessorTime;
        processCpuStruct.lastSampleTime = now;

        var cpuUsage = cpuUsed.TotalMilliseconds / (elapsed.TotalMilliseconds * Environment.ProcessorCount) * 100f;
        return (float)cpuUsage;
    }

    private float GetTotalMemoryUsage(ProcessStruct processStruct)
    {
        long memBytes = processStruct.process.WorkingSet64;
        float memMB = memBytes / (1024f * 1024f);
        return memMB;
    }

    public void LogPerformanceData()
    {
        LogData("Performance/gameCpuUsage(%)", GetCpuUsage(mainProcessStruct));
        LogData("Performance/gameMemoryUsage(MB)", GetTotalMemoryUsage(mainProcessStruct));
    }

    public void LogEpisodeTime()
    {
        DateTime logTime = DateTime.Now;
        LogData("Performance/episodeLength(s)", (float)(logTime - lastEpisodeMeasure).TotalSeconds);
        lastEpisodeMeasure = logTime;
    }
    
    public void LogStepTime()
    {
        DateTime logTime = DateTime.Now;
        LogData("Performance/stepLength(s)", (float)(logTime - lastStepMeasure).TotalSeconds/frequency);
        lastStepMeasure = logTime;
    }

    public void LogData(string name, float value)
    {
        recorder.Add(name, value, StatAggregationMethod.Average);
    }

    private class ProcessStruct
    {
        public TimeSpan lastCpuTime;
        public DateTime lastSampleTime;
        public Process process;

        public ProcessStruct(Process process)
        {
            this.process = process;
            this.lastCpuTime = process.TotalProcessorTime;
            this.lastSampleTime = DateTime.UtcNow;
        }
    }
}