import { MenuBarDemo } from "@/components/ui/demo";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-6 bg-[#0B0D11] text-white">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-orange-400 to-amber-200 bg-clip-text text-transparent">
          SignBridge UI
        </h1>
        <p className="text-sm text-zinc-400 mt-2">
          Interactive Dynamic Island Bottom Menu Bar
        </p>
      </div>

      <div className="w-full max-w-md p-8 rounded-2xl bg-zinc-900/60 border border-zinc-800 shadow-2xl backdrop-blur-xl flex flex-col items-center justify-center min-h-[300px]">
        <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-8">
          Hover icons to inspect spring tooltip & glide physics
        </p>
        <MenuBarDemo />
      </div>
    </div>
  );
}
