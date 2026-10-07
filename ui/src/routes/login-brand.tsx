import "./login.css";
import contractorLogoUrl from "../assets/contractor-logo.png";

/** The product mark, name and line on sign-in and connection pages. */
export function SignInBrand() {
  return (
    <div className="ops-signin-brand">
      <img
        className="ops-signin-mark"
        src={contractorLogoUrl}
        alt=""
        aria-hidden="true"
      />
      <div>
        <p className="ops-signin-product">Contractor</p>
        <p className="ops-signin-tagline">Security research &amp; automation</p>
      </div>
    </div>
  );
}
