
using System;
using System.Runtime.InteropServices;
using System.Threading;

public class Example{
    private const int HOST_CPU_LOAD_INFO = 3;
    private const int CPU_STATE_MAX = 4;
    private const int CPU_STATE_USER = 0;
    private const int CPU_STATE_SYSTEM = 1;
    private const int CPU_STATE_IDLE = 2;
    private const int CPU_STATE_NICE = 3;

    [DllImport("libc")]
    private static extern IntPtr mach_host_self();

    [DllImport("libc")]
    private static extern int host_statistics(
        IntPtr host,
        int flavor,
        IntPtr host_info,
        ref int host_info_count
    );

    [StructLayout(LayoutKind.Sequential)]
    private struct CpuLoadInfo
    {
        public uint user;
        public uint system;
        public uint idle;
        public uint nice;
    }

    private static CpuLoadInfo previousLoad = new CpuLoadInfo();

    public static double GetCpuUsage()
    {
        IntPtr host = mach_host_self();
        int infoCount = 4; // CPU_STATE_MAX
        
        CpuLoadInfo loadInfo = new CpuLoadInfo();
        IntPtr loadInfoPtr = Marshal.AllocHGlobal(Marshal.SizeOf(loadInfo));
        
        try
        {
            int result = host_statistics(host, HOST_CPU_LOAD_INFO, loadInfoPtr, ref infoCount);
            
            if (result != 0)
            {
                return -1; // Error
            }

            loadInfo = Marshal.PtrToStructure<CpuLoadInfo>(loadInfoPtr);

            // Calculate deltas
            uint userDelta = loadInfo.user - previousLoad.user;
            uint systemDelta = loadInfo.system - previousLoad.system;
            uint idleDelta = loadInfo.idle - previousLoad.idle;
            uint niceDelta = loadInfo.nice - previousLoad.nice;

            uint totalDelta = userDelta + systemDelta + idleDelta + niceDelta;

            // Store current values for next calculation
            previousLoad = loadInfo;

            if (totalDelta == 0)
            {
                return 0;
            }

            // Calculate CPU usage percentage (excluding idle time)
            double usedDelta = userDelta + systemDelta + niceDelta;
            return (usedDelta / totalDelta) * 100.0;
        }
        finally
        {
            Marshal.FreeHGlobal(loadInfoPtr);
        }
    }

    public static void Main()
    {
        // Initialize - first call to establish baseline
        GetCpuUsage();
        Thread.Sleep(1000);

        // Now we can poll frequently
        for (int i = 0; i < 20; i++)
        {
            double cpuUsage = GetCpuUsage();
            Console.WriteLine($"CPU Usage: {cpuUsage:F2}%");
            
            Thread.Sleep(1000); // Can poll every 50ms with minimal overhead!
        }
    }
}
// Source - https://stackoverflow.com/a
// Posted by Christian C. Salvadó, modified by community. See post 'Timeline' for change history
// Retrieved 2025-11-13, License - CC BY-SA 4.0

//	public string getCurrentCpuUsage(){
  //   	return cpuCounter.NextValue()+"%";
	//}	

	//public string getAvailableRAM(){
     //   return ramCounter.NextValue()+"MB";
//	} 


