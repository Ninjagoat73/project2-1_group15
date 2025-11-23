using System;
using System.Diagnostics;
using Unity.MLAgents;

public class PerformanceLogger
{
    private ProcessStruct mainProcessStruct;
    private DateTime lastStepMeasure;
    private DateTime lastEpisodeMeasure;
    private StatsRecorder recorder;
    private int frequency;
    private DateTime startTime;

    /// <summary>
    //  Class used for logging data from c# level. Currently we are logging game's cpu usage, game's memory usage
    //  time per step, steps per second,time per episode. Example of usage alongside with commants explaining 
    //  which lines to add can be found in Project/Assets/ML-Agents/Examples/3DBall/Scripts/Ball3DAgent.cs
    //  How to use in game's agent script (Ball3DAgent.cs, FoodCollector.cs, GridAgent.cs etc.):
    //  Add follwing lines in class attribute fields:
    //  -- at the end of Initialize add line: base.InitializeLoggingVariables();
    //  -- at the beginning of OnActionReceived() add line: base.CheckAndLogFirstPart();
    //  -- at the end of OnActionReceived() add line: base.ManageStepCount();
    //  -- at the end of OnEpisodeBegin() add line: base.LogEpisodeTime();
    /// </summary>
    /// <param name="recorder">StatsRecorder object associated with current training, used for logging data into tensorboard</param>
    /// <param name="mainProcess">Process object associated with the running unity game</param>
    /// <param name="frequency">Frequency of logging data</param>
    public PerformanceLogger(StatsRecorder recorder, Process mainProcess, int frequency)
    {
        this.mainProcessStruct = new ProcessStruct(mainProcess);
        this.recorder = recorder;
        this.lastStepMeasure = DateTime.Now;
        this.lastEpisodeMeasure = DateTime.Now;
        this.startTime = DateTime.Now;
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

    // Logs perfromacne data regarding current unity game
    public void LogPerformanceData()
    {
        LogData("Performance/gameCpuUsage(%)", GetCpuUsage(mainProcessStruct));
        LogData("Performance/gameMemoryUsage(MB)", GetTotalMemoryUsage(mainProcessStruct));
    }

    // Logs total time elapsed from the beginning of training
    public void LogTimeElapsed()
    {
        recorder.Add("Performance/timeElapsed(s)", (float)(DateTime.Now - startTime).TotalSeconds, StatAggregationMethod.MostRecent);
    }

    // Logs last episode's time in seconds
    public void LogEpisodeTime()
    {
        DateTime logTime = DateTime.Now;
        LogData("Performance/episodeTime(s)", (float)(logTime - lastEpisodeMeasure).TotalSeconds);
        lastEpisodeMeasure = logTime;
    }

    // Logs average time per step and average steps per second
    public void LogStepTimeAndStepPerSecond()
    {
        DateTime logTime = DateTime.Now;
        float diff = (float)(logTime - lastStepMeasure).TotalSeconds / frequency;
        LogData("Performance/stepLength(ms)", diff * 1000);
        LogData("Performance/stepsPerSecond", (float)frequency / diff);
        lastStepMeasure = logTime;
    }

    // Helper method used for logging the data
    public void LogData(string name, float value)
    {
        recorder.Add(name, value, StatAggregationMethod.Average);
    }

    // Helper class used for keeping track of Process data used
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