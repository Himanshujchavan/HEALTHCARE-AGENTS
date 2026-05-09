import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { useRegister } from "../api/hooks";
import { UserCreate } from "../models/auth";

export function RegisterPage() {
  const navigate = useNavigate();
  const { register, handleSubmit } = useForm<UserCreate>();
  const { mutateAsync, isPending, error } = useRegister();

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#ecfdf3_0%,_transparent_45%),linear-gradient(135deg,_#fdfcfb_0%,_#f7efe5_100%)]">
      <div className="mx-auto flex max-w-md flex-col gap-6 px-6 py-20">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Get started</p>
          <h1 className="text-3xl font-semibold text-slate-900">Create account</h1>
        </div>

        <form
          className="space-y-4 rounded-3xl border border-slate-200 bg-white/80 p-6"
          onSubmit={handleSubmit(async (values) => {
            await mutateAsync(values);
            navigate("/login");
          })}
        >
          <label className="space-y-2 text-sm text-slate-600">
            Username
            <input
              type="text"
              className="w-full rounded-xl border border-slate-200 px-4 py-3"
              {...register("username")}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-600">
            Email
            <input
              type="email"
              className="w-full rounded-xl border border-slate-200 px-4 py-3"
              {...register("email")}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-600">
            Password
            <input
              type="password"
              className="w-full rounded-xl border border-slate-200 px-4 py-3"
              {...register("password")}
            />
          </label>

          {error && <p className="text-sm text-red-600">Registration failed.</p>}

          <button
            type="submit"
            disabled={isPending}
            className="w-full rounded-full bg-emerald-600 px-6 py-3 text-sm font-semibold text-white"
          >
            {isPending ? "Creating..." : "Register"}
          </button>
        </form>
      </div>
    </div>
  );
}
