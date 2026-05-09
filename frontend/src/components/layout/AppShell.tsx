import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

export function AppShell() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#d1fae5_0%,_transparent_45%),linear-gradient(120deg,_#fdfcfb_0%,_#f7f0e7_100%)]">
      <div className="mx-auto flex min-h-screen max-w-[1600px]">
        <Sidebar />
        <div className="flex flex-1 flex-col">
          <TopBar />
          <main className="flex-1 px-8 py-8">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
}
